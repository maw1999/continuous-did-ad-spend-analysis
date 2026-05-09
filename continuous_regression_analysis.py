import pandas as pd
import scipy.stats as stats
from datetime import datetime
import json
import re
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Load the table data
def get_state_age_map(file_path):
	df = pd.read_csv(file_path, header=None)

	data = df.iloc[4:316, [0, 1, 3]].copy()
	data.columns = ['State', 'Age Bucket', 'Total Citizen Population']

	data['State'] = data['State'].ffill().str.replace('[merged] ', '', regex=False).str.strip()

	def clean_numeric(val):
		if pd.isna(val):
			return 0
		clean_val = str(val).replace(',', '').strip()
		try:
			# Total citizen population is already in thousands.
			return float(clean_val) / 100.0
		except ValueError:
			return 0

	data['Total Citizen Population (100,000s)'] = data['Total Citizen Population'].apply(clean_numeric)

	state_age_map = {}

	for _, row in data.iterrows():
		state = row['State'].lower()
		age_bucket = str(row['Age Bucket']).strip()
		population = row['Total Citizen Population (100,000s)']
		
		if state not in state_age_map:
			state_age_map[state] = {}

		state_age_map[state][age_bucket] = population

	return state_age_map

# Merged function that extracts the micro-windows dynamically AND aggregates campaign data
def extract_micro_windows_and_campaigns(file_path):
	df = pd.read_csv(file_path)

	def clean_pct(val):
		if pd.isna(val) or val == '': return 0.0
		try:
			return float(str(val).replace('%', '')) / 100.0
		except:
			return 0.0

	def clean_spend(val):
		if pd.isna(val) or val == '': return 0
		try:
			return int(re.sub(r'[^\d]', '', str(val)))
		except:
			return 0

	age_mapping = {
		"18-24": "18 to 24 years",
		"25-34": "25 to 34 years",
		"35-44": "35 to 44 years",
		"45-54": "45 to 64 years",
		"55-64": "45 to 64 years",
		"65+": "65 years and over"
	}

	df['California Delivery Num'] = df['California Delivery'].apply(clean_pct)
	df['Non-California Delivery Num'] = df['Non-California Delivery'].apply(clean_pct)
	df['lower_bound_spend_num'] = df['lower_bound_spend'].apply(clean_spend)
	
	target_keywords = ['donate', 'chip in', 'contribute', 'support']
	def contains_target_keywords(text):
		if pd.isna(text):
			return False
		text_lower = str(text).lower()
		return any(keyword in text_lower for keyword in target_keywords)

	# Check for Non-California Delivery > 50%
	campaign_ads = df[df['Non-California Delivery Num'] > 0.5].copy()
	
	# Apply the ads content filter across relevant columns
	content_mask = (
		campaign_ads['ad_creative_bodies'].apply(contains_target_keywords) |
		campaign_ads['ad_creative_link_titles'].apply(contains_target_keywords) |
		campaign_ads['ad_creative_link_descriptions'].apply(contains_target_keywords)
	)
	
	campaign_ads = campaign_ads[content_mask]

	# Establish exact dates to distribute daily spend
	campaign_ads['start'] = pd.to_datetime(campaign_ads['ad_delivery_start_time'], errors='coerce').dt.normalize()
	campaign_ads['stop'] = pd.to_datetime(campaign_ads['ad_delivery_stop_time'], errors='coerce').dt.normalize()
	campaign_ads = campaign_ads.dropna(subset=['start'])

	max_date = campaign_ads['stop'].max()
	if pd.isna(max_date): max_date = pd.Timestamp.today().normalize()
	campaign_ads['stop'] = campaign_ads['stop'].fillna(max_date)

	campaign_ads['duration'] = (campaign_ads['stop'] - campaign_ads['start']).dt.days + 1
	campaign_ads['daily_spend'] = campaign_ads['lower_bound_spend_num'] / campaign_ads['duration']

	if campaign_ads.empty:
		return {}, []

	date_range = pd.date_range(start=campaign_ads['start'].min(), end=campaign_ads['stop'].max())
	daily_spend = pd.Series(0.0, index=date_range)

	for _, row in campaign_ads.iterrows():
		if row['duration'] > 0:
			daily_spend.loc[row['start']:row['stop']] += row['daily_spend']

	# 2. The Activation Threshold (Continuous Block)
	active_days = daily_spend > 0
	blocks = []
	current_start = None

	for date, is_active in active_days.items():
		if is_active and current_start is None:
			current_start = date
		elif not is_active and current_start is not None:
			blocks.append((current_start, date - pd.Timedelta(days=1)))
			current_start = None
	if current_start is not None:
		blocks.append((current_start, active_days.index[-1]))

	raw_periods = []
	last_active_end = pd.Timestamp.min

	for start, stop in blocks:
		duration = (stop - start).days + 1
		total_spend = daily_spend.loc[start:stop].sum()

		# Spike Check Threshold
		if total_spend >= 5000:
			# Determine baseline period symmetry
			base_start = start - pd.Timedelta(days=duration)
			base_end = start - pd.Timedelta(days=1)

			# Ensure window is isoldated
			if base_start > last_active_end:
				raw_periods.append({
					'base_start': base_start.strftime('%Y-%m-%d'),
					'base_end': base_end.strftime('%Y-%m-%d'),
					'act_start': start.strftime('%Y-%m-%d'),
					'act_end': stop.strftime('%Y-%m-%d')
				})
				last_active_end = stop

	campaign_dict = {}
	valid_periods = []

	# Construct Final Campaigns for Valid Periods
	for p in raw_periods:
		start = pd.to_datetime(p['act_start'])
		stop = pd.to_datetime(p['act_end'])

		mask = (campaign_ads['start'] <= stop) & (campaign_ads['stop'] >= start)
		period_ads = campaign_ads[mask]

		if period_ads.empty:
			continue

		total_lower_spend = period_ads['lower_bound_spend_num'].sum()

		if total_lower_spend > 0:
			weighted_ca_pct = (period_ads['California Delivery Num'] * period_ads['lower_bound_spend_num']).sum() / total_lower_spend
			weighted_non_ca_pct = (period_ads['Non-California Delivery Num'] * period_ads['lower_bound_spend_num']).sum() / total_lower_spend
		else:
			weighted_ca_pct = 0.0
			weighted_non_ca_pct = 0.0

		age_cohorts = {}
		regions = {}

		for _, row in period_ads.iterrows():
			ad_spend = row.get('lower_bound_spend_num', 0)

			dist_str = row.get('demographic_distribution')
			# Compute demographic distribution weighted by ad spend
			if pd.notna(dist_str):
				try:
					items = json.loads("[" + str(dist_str) + "]")
					ad_age_pct = {}
					for item in items:
						raw_age = item.get('age')
						pct = float(item.get('percentage', 0))
						if raw_age:
							mapped_age = age_mapping.get(raw_age, raw_age)
							ad_age_pct[mapped_age] = ad_age_pct.get(mapped_age, 0.0) + pct

					for age, total_pct in ad_age_pct.items():
						age_cohorts[age] = age_cohorts.get(age, 0.0) + (ad_spend * total_pct)
				except Exception:
					pass

			# Compute delivery by region weighted by ad spend
			reg_str = row.get('delivery_by_region')
			if pd.notna(reg_str):
				try:
					items = json.loads("[" + str(reg_str) + "]")
					for item in items:
						reg = item.get('region')
						pct = float(item.get('percentage', 0))
						regions[reg] = regions.get(reg, 0.0) + (ad_spend * pct)
				except:
					pass

		if total_lower_spend > 0:
			age_cohorts = {k: v / total_lower_spend for k, v in age_cohorts.items()}
			regions = {k: v / total_lower_spend for k, v in regions.items()}

		key = (p['act_start'], p['act_end'])
		
		# Adding page_ids, creation_ids, and ad_creative_bodies alongside calculated pcts
		campaign_dict[key] = {
			"page_ids": period_ads['page_id'].unique().tolist(),
			"creation_ids": period_ads['ad_creation_time'].unique().tolist(),
			"ad_creative_bodies": period_ads['ad_creative_bodies'].dropna().unique().tolist(),
			"start_date": key[0],
			"stop_date": key[1],
			"aggregate_demographic_distribution": age_cohorts,
			"aggregate_delivery_by_region": regions,
			"total_lower_spend": total_lower_spend,
			"aggregate_california_delivery": f"{weighted_ca_pct:.2%}",
			"aggregate_non_california_delivery": f"{weighted_non_ca_pct:.2%}",
		}
		valid_periods.append(p)

	return campaign_dict, valid_periods
	
def print_top_spend_regions(campaign_dict, campaign_key, top_x=3):
	if campaign_key not in campaign_dict:
		print(f"Error: Campaign key {campaign_key} not found.")
		return None

	regions = campaign_dict[campaign_key].get("aggregate_delivery_by_region", {})
	if not regions:
		return None

	sorted_regions = sorted(regions.items(), key=lambda item: item[1], reverse=True)
	top_regions = sorted_regions[:top_x]

	print(f"Top {top_x} highest spend regions for {campaign_key}:")
	for i, (region, value) in enumerate(top_regions, 1):
		print(f"\t{i}. {region}: {value:.2%} of total spend")
	
	return top_regions
	
def print_top_spend_age_cohorts(campaign_dict, campaign_key, top_x=3):
	if campaign_key not in campaign_dict:
		print(f"Error: Campaign key {campaign_key} not found.")
		return None

	age_cohorts = campaign_dict[campaign_key].get("aggregate_demographic_distribution", {})
	if not age_cohorts:
		return None

	sorted_cohorts = sorted(age_cohorts.items(), key=lambda item: item[1], reverse=True)
	top_cohorts = sorted_cohorts[:top_x]

	print(f"Top {top_x} highest spend age cohorts for {campaign_key}:")
	for i, (age, value) in enumerate(top_cohorts, 1):
		print(f"\t{i}. Age {age}: {value:.2%} of total spend")
	
	return top_cohorts


def run_continuous_dose_response_model(fec_file, state_age_map, ad_campaigns, periods):
	# California is excluded as it is the candidate's home state
	state_abbr_to_name = {
		'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
		'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
		'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
		'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
		'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
		'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire',
		'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina',
		'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania',
		'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee',
		'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
		'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
	}

	valid_buckets = list(state_abbr_to_name.values()) 

	fec_df = pd.read_csv(fec_file)
	fec_df = fec_df[fec_df['entity_type'] == 'IND'].copy()
	fec_df = fec_df.dropna(subset=['contribution_receipt_date', 'contributor_name', 'contributor_city', 'contributor_state', 'contributor_occupation'])
	fec_df['contribution_receipt_date'] = pd.to_datetime(fec_df['contribution_receipt_date'], errors='coerce')

	def assign_bucket(row):
		state_abbr = str(row['contributor_state']).upper()
		state_name = state_abbr_to_name.get(state_abbr, None)
		return state_name

	fec_df['bucket'] = fec_df.apply(assign_bucket, axis=1)
	fec_df = fec_df.dropna(subset=['bucket'])
	
	fec_df['contributor_id'] = fec_df[['contributor_name', 'contributor_city', 'contributor_state', 'contributor_occupation']].astype(str).apply(lambda x: '|'.join(x), axis=1)

	def get_campaign_metrics_for_period(act_start, act_end, campaigns):
		key = (act_start.strftime('%Y-%m-%d'), act_end.strftime('%Y-%m-%d'))
		if key in campaigns:
			data = campaigns[key]
			spend = data['total_lower_spend']
			age_distribution_pct = data['aggregate_demographic_distribution']
			region_delivery_dollars = {
				reg: pct * spend for reg, pct in data['aggregate_delivery_by_region'].items()
			}
			return spend, age_distribution_pct, region_delivery_dollars, data["total_lower_spend"]
		return 0, {}, {}

	def get_tam(bucket, active_age_cohorts, state_map):
		pop = 0
		state_data = state_map.get(bucket.lower(), {})
		for age in active_age_cohorts:
			pop += state_data.get(age, 0)
		return pop

	def get_region_delivery(bucket, region_delivery_dollars):
		return region_delivery_dollars.get(bucket, 0)

	# DIFFERENCE-IN-DIFFERENCES: We no longer use simple X and Y arrays. 
	# We construct a panel dataset (list of dicts) to feed into a DataFrame.
	panel_data = []
	aggregate_ad_lower_spend = 0

	for period_idx, period in enumerate(periods):
		act_start = pd.to_datetime(period['act_start'])
		act_end = pd.to_datetime(period['act_end'])
		base_start = pd.to_datetime(period['base_start'])
		base_end = pd.to_datetime(period['base_end'])
		
		act_days = (act_end - act_start).days + 1
		base_days = (base_end - base_start).days + 1
		
		period_spend, age_dist_pct, region_delivery_dollars, total_ad_spend = get_campaign_metrics_for_period(act_start, act_end, ad_campaigns)
		aggregate_ad_lower_spend += total_ad_spend
		active_cohorts = [age for age, pct in age_dist_pct.items() if pct >= 0.05]
		
		for bucket in valid_buckets:
			tam = get_tam(bucket, active_cohorts, state_age_map)
			if tam == 0:
				continue
				
			dose = get_region_delivery(bucket, region_delivery_dollars)
			dose_per_capita = dose / tam
			
			bucket_df = fec_df[fec_df['bucket'] == bucket]
			
			# Calculate absolute daily yields instead of net marginal yields
			base_mask = (bucket_df['contribution_receipt_date'] >= base_start) & (bucket_df['contribution_receipt_date'] <= base_end)
			yield_baseline = (bucket_df[base_mask]['contributor_id'].nunique() / base_days) / tam
			
			act_mask = (bucket_df['contribution_receipt_date'] >= act_start) & (bucket_df['contribution_receipt_date'] <= act_end)
			yield_active = (bucket_df[act_mask]['contributor_id'].nunique() / act_days) / tam
			
			# DIFFERENCE-IN-DIFFERENCES: Append two separate rows per state, per period.
			# 1. The Baseline Row (is_active = 0)
			panel_data.append({
				'state': bucket,
				'period_id': period_idx, # Used as a fixed effect to control for election proximity
				'is_active': 0,
				'dose_per_capita': dose_per_capita, # Dose applied here to control for geographic selection bias
				'daily_yield_per_capita': yield_baseline,
				'net_delta': yield_active - yield_baseline # Stored only for 2D visualization later
			})
			
			# 2. The Active Row (is_active = 1)
			panel_data.append({
				'state': bucket,
				'period_id': period_idx,
				'is_active': 1,
				'dose_per_capita': dose_per_capita,
				'daily_yield_per_capita': yield_active,
				'net_delta': yield_active - yield_baseline
			})

	df_panel = pd.DataFrame(panel_data)

	# DIFFERENCE-IN-DIFFERENCES: The Regression Model
	# Equation: Yield = is_active + dose_per_capita + (is_active * dose_per_capita) + period_fixed_effects
	# In patsy/statsmodels syntax, '*' automatically includes the main effects and the interaction.
	# C(period_id) treats the periods as categorical variables to normalize time acceleration.
	formula = 'daily_yield_per_capita ~ is_active * dose_per_capita + C(period_id)'
	
	model = smf.ols(formula, data=df_panel)
	
	# We cluster standard errors by 'state' to correct for panel autocorrelation
	results = model.fit(cov_type='cluster', cov_kwds={'groups': df_panel['state']})

	print("Total ad spend across all ad active windows: " + str(aggregate_ad_lower_spend))
	print("\n--- CONTINUOUS DIFFERENCE-IN-DIFFERENCES RESULTS ---")
	print(results.summary())
	print("----------------------------------------------------------\n")

	# Extract specific coefficients for our visualization
	# 'is_active:dose_per_capita' is our true isolated ROI (the slope)
	# 'is_active' is the exogenous temporal shock (the intercept shift)
	interaction_coef = results.params.get('is_active:dose_per_capita', 0)
	time_shock_coef = results.params.get('is_active', 0)
	p_val_interaction = results.pvalues.get('is_active:dose_per_capita', 1.0)

	return {
		'df_panel': df_panel,
		'interaction_coef': interaction_coef,
		'time_shock_coef': time_shock_coef,
		'p_value': p_val_interaction,
		'r_squared': results.rsquared
	}

def visualize_dose_response(results):
	df = results['df_panel']
	
	# For a 2D scatter plot, we plot the "First Differences" (Net Delta).
	# We only need one row per state-period to plot the points.
	df_plot = df[df['is_active'] == 1].copy()
	
	X = df_plot['dose_per_capita']
	Y = df_plot['net_delta']
	
	slope = results['interaction_coef']
	intercept = results['time_shock_coef'] 
	r_squared = results['r_squared']
	p_value = results['p_value']

	plt.figure(figsize=(10, 6))
	plt.scatter(X, Y, color='#2c3e50', alpha=0.6, label='Geotemporal Micro-Windows')

	if not X.empty: 
		x_min, x_max = min(X), max(X)
		x_line = [x_min, x_max]
		# The DiD formula translates to: Delta Y = Time Shock (Intercept) + True ROAS (Slope) * Dose
		y_line = [slope * x + intercept for x in x_line]
		
		plt.plot(
			x_line, y_line, 
			color='#e74c3c', 
			linewidth=2, 
			label=f'DiD Isolated Effect\nYield = {slope:.4f}(Dose) + {intercept:.4f}'
		)

	plt.title('Continuous DiD: Causal Effect of Meta Spend on Donor Yield', fontsize=14, fontweight='bold')
	plt.xlabel('Empirical Financial Dose Per Capita (Absolute Spend / TAM)', fontsize=12)
	plt.ylabel('First Difference: Net New Donors Per Capita ($\Delta$ Yield)', fontsize=12)

	stats_text = f'Interaction p-value = {p_value:.2e}\nPanel R² = {r_squared:.4f}'
	plt.text(
		0.05, 0.95, 
		stats_text, 
		transform=plt.gca().transAxes, 
		fontsize=12,
		verticalalignment='top', 
		bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#bdc3c7', alpha=0.9)
	)

	plt.grid(True, linestyle='--', alpha=0.5)
	plt.legend(loc='lower right')
	plt.tight_layout()
	plt.show()

# MAIN EXECUTION
if __name__ == "__main__":
	state_age_map = get_state_age_map('voters_by_age_by_state_original.csv')
	
	# Dynamically extract the overlapping periods and compile campaigns
	ad_campaigns, periods = extract_micro_windows_and_campaigns('meta_ad_library.csv')
	
	print(f"Extracted {len(periods)} valid micro-windows:")
	for p in periods:
		print(f"\tBaseline: {p['base_start']} to {p['base_end']} -> Active: {p['act_start']} to {p['act_end']}")

	# Pass both the dynamic dictionary and the dynamic periods list into the model
	regression_results = run_continuous_dose_response_model('fec_contributions_cross_ref.csv', state_age_map, ad_campaigns, periods)

	visualize_dose_response(regression_results)
