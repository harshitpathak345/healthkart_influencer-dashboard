import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker with Indian locale for names
fake = Faker('en_IN')

# Set a seed for reproducibility
np.random.seed(42)
random.seed(42)

# --- 1. Simulate influencers data ---
def simulate_influencers(num_influencers=100): # Using original size, but start smaller if memory is still an issue
    influencers = []
    platforms = ['Instagram', 'YouTube', 'Twitter']
    categories = ['Fitness', 'Nutrition', 'Lifestyle', 'Beauty', 'Health', 'Cooking']
    genders = ['Male', 'Female', 'Other']

    for i in range(num_influencers):
        _id = f'INF{i+1:03d}'
        name = fake.name()
        category = random.choice(categories)
        gender = random.choice(genders)
        
        # Follower count distribution: more smaller influencers, fewer very large ones
        if random.random() < 0.6: # 60% are micro/small influencers
            follower_count = random.randint(5_000, 100_000)
        elif random.random() < 0.9: # 30% are mid-tier
            follower_count = random.randint(100_001, 1_000_000)
        else: # 10% are macro
            follower_count = random.randint(1_000_001, 5_000_000)
            
        platform = random.choice(platforms)
        influencers.append([_id, name, category, gender, follower_count, platform])

    df_influencers = pd.DataFrame(influencers, columns=['ID', 'name', 'category', 'gender', 'follower_count', 'platform'])
    return df_influencers

# --- 2. Simulate posts data ---
def simulate_posts(df_influencers, num_posts=1500): # Using original size
    posts = []
    start_date = datetime(2024, 6, 1) # Adjusted start date to match screenshot's time frame
    end_date = datetime(2024, 7, 21) # Adjusted end date to match screenshot's time frame

    influencer_platform_map = df_influencers.set_index('ID')['platform'].to_dict()
    influencer_follower_map = df_influencers.set_index('ID')['follower_count'].to_dict()

    valid_influencer_ids = list(influencer_platform_map.keys())

    if not valid_influencer_ids:
        print("Warning: No valid influencers found to simulate posts.")
        return pd.DataFrame(columns=['influencer_id', 'platform', 'date', 'URL', 'caption', 'reach', 'likes', 'comments'])

    for i in range(num_posts):
        influencer_id = random.choice(valid_influencer_ids)
        platform = influencer_platform_map[influencer_id]
        date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        url = fake.url()
        caption = fake.sentence(nb_words=random.randint(10, 30))

        follower_count = influencer_follower_map[influencer_id]
        reach = int(follower_count * random.uniform(0.1, 0.6))
        likes = int(reach * random.uniform(0.05, 0.2)) # 5-20% of reach
        comments = int(likes * random.uniform(0.05, 0.15)) # 5-15% of likes

        posts.append([influencer_id, platform, date.strftime('%Y-%m-%d'), url, caption, reach, likes, comments])

    df_posts = pd.DataFrame(posts, columns=['influencer_id', 'platform', 'date', 'URL', 'caption', 'reach', 'likes', 'comments'])
    return df_posts

# --- 3. Simulate tracking_data ---
def simulate_tracking_data(df_influencers, num_tracking_entries=7000): # Using original size
    tracking_data = []
    start_date = datetime(2024, 6, 1) # Adjusted start date
    end_date = datetime(2024, 7, 21) # Adjusted end date
    
    # Specific HealthKart brands and their products
    products_brands = {
        'Whey Protein (MB)': 'MuscleBlaze', 'Creatine (MB)': 'MuscleBlaze', 'BCAA (MB)': 'MuscleBlaze',
        'Multivitamin (HK)': 'HKVitals', 'Fish Oil (HK)': 'HKVitals', 'Biotin (HK)': 'HKVitals',
        'Kids Nutrition (GZ)': 'Gritzo', 'Kids Immunity (GZ)': 'Gritzo'
    }
    product_list = list(products_brands.keys())

    influencer_ids = df_influencers['ID'].tolist()

    if not influencer_ids:
        print("Warning: No valid influencers found to simulate tracking data.")
        return pd.DataFrame(columns=['source', 'campaign', 'influencer_id', 'user_id', 'product', 'date', 'orders', 'revenue', 'brand'])

    for i in range(num_tracking_entries):
        source = 'Influencer Marketing'
        campaign = random.choice(['Summer Sale', 'Winter Wellness', 'Festive Offer', 'New Product Launch'])
        influencer_id = random.choice(influencer_ids)
        user_id = f'USR{i+1:05d}'
        product = random.choice(product_list)
        date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        orders = 1 # Assuming one order per tracking entry for simplicity
        revenue = round(random.uniform(500, 7000), 2) # Adjusted revenue range to match screenshot L values

        tracking_data.append([source, campaign, influencer_id, user_id, product, date.strftime('%Y-%m-%d'), orders, revenue])

    df_tracking_data = pd.DataFrame(tracking_data, columns=['source', 'campaign', 'influencer_id', 'user_id', 'product', 'date', 'orders', 'revenue'])
    df_tracking_data['brand'] = df_tracking_data['product'].map(products_brands)
    return df_tracking_data

# --- 4. Simulate payouts data ---
def simulate_payouts(df_influencers, df_posts, df_tracking_data):
    payouts = []
    influencer_ids = df_influencers['ID'].tolist()

    for influencer_id in influencer_ids:
        basis = random.choice(['post', 'order'])

        if basis == 'post':
            num_posts = df_posts[df_posts['influencer_id'] == influencer_id].shape[0]
            if num_posts > 0:
                rate = round(random.uniform(1000, 30000), 2) # Rate per post, adjusted for typical payouts
                total_payout = rate * num_posts
                payouts.append([influencer_id, basis, rate, None, total_payout])
        else: # basis == 'order'
            attributed_orders = df_tracking_data[df_tracking_data['influencer_id'] == influencer_id]['orders'].sum()
            if attributed_orders > 0:
                rate = round(random.uniform(50, 300), 2) # Rate per order, adjusted
                total_payout = rate * attributed_orders
                payouts.append([influencer_id, basis, rate, attributed_orders, total_payout])

    df_payouts = pd.DataFrame(payouts, columns=['influencer_id', 'basis', 'rate', 'orders', 'total_payout'])
    return df_payouts

# Generate and save data
if __name__ == "__main__":
    # For initial testing with Parquet, you might want to use smaller numbers first
    # e.g., num_influencers=20, num_posts=200, num_tracking_entries=500
    # Once confirmed working, you can change them back to the original larger values.
    
    df_influencers = simulate_influencers(num_influencers=100) # Use 100 for final version
    df_influencers.to_parquet('influencers.parquet', index=False) # CHANGED TO PARQUET
    print("Generated influencers.parquet")

    df_posts = simulate_posts(df_influencers, num_posts=1500) # Use 1500 for final version
    df_posts.to_parquet('posts.parquet', index=False) # CHANGED TO PARQUET
    print("Generated posts.parquet")

    df_tracking_data = simulate_tracking_data(df_influencers, num_tracking_entries=7000) # Use 7000 for final version
    df_tracking_data.to_parquet('tracking_data.parquet', index=False) # CHANGED TO PARQUET
    print("Generated tracking_data.parquet")

    df_payouts = simulate_payouts(df_influencers, df_posts, df_tracking_data) # CHANGED TO PARQUET
    df_payouts.to_parquet('payouts.parquet', index=False) # CHANGED TO PARQUET
    print("Generated payouts.parquet")

    print("\nAll simulated data files created successfully.")