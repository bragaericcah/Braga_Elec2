import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, when, round as spark_round
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load data using pandas first"""
    print("Loading data...")
    # Load delays data
    delays_df = pd.read_csv("departuredelays.csv")
    print(f"Loaded {len(delays_df):,} flight records")
    
    # Load airport data
    airports_df = pd.read_csv("airport-codes-na.txt", sep='\t')
    print(f"Loaded {len(airports_df):,} airport records")
    
    print("\nFirst 10 rows of departure delays data:")
    print(delays_df.head(10))
    
    print("\nFirst 10 rows of airport codes data:")
    print(airports_df.head(10))
    
    return delays_df, airports_df

def analyze_airport_delays(delays_df, airports_df):
    """Analyze delays by airport with location information"""
    # Merge delays with airport information
    airport_delays = delays_df.merge(
        airports_df[['IATA', 'City', 'State']],
        left_on='origin',
        right_on='IATA',
        how='left'
    )
    
    print("\nFirst 10 rows of joined data:")
    print(airport_delays[['date', 'delay', 'distance', 'origin', 'City', 'State']].head(10))
    
    # Calculate airport statistics
    stats = []
    for (origin, city, state), group in airport_delays.groupby(['origin', 'City', 'State']):
        total_flights = len(group)
        avg_delay = group['delay'].mean()
        delay_std = group['delay'].std()
        avg_distance = group['distance'].mean()
        significant_delays = (group['delay'] > 15).sum()
        delay_ratio = significant_delays / total_flights
        
        stats.append({
            'origin': origin,
            'city': city,
            'state': state,
            'total_flights': total_flights,
            'avg_delay': round(avg_delay, 2),
            'delay_std': round(delay_std, 2),
            'avg_distance': round(avg_distance, 2),
            'significant_delays': significant_delays,
            'delay_ratio': round(delay_ratio, 3)
        })
    
    airport_stats = pd.DataFrame(stats)
    airport_stats['location'] = airport_stats['city'] + ', ' + airport_stats['state']
    
    return airport_stats.sort_values('avg_delay', ascending=False)

def analyze_routes(delays_df, airports_df):
    """Analyze route patterns with full airport names"""
    # Merge with origin airport info
    routes = delays_df.merge(
        airports_df[['IATA', 'City', 'State']].rename(
            columns={'IATA': 'orig_code', 'City': 'origin_city', 'State': 'origin_state'}
        ),
        left_on='origin',
        right_on='orig_code'
    )
    
    # Merge with destination airport info
    routes = routes.merge(
        airports_df[['IATA', 'City', 'State']].rename(
            columns={'IATA': 'dest_code', 'City': 'dest_city', 'State': 'dest_state'}
        ),
        left_on='destination',
        right_on='dest_code'
    )
    
    # Create full names
    routes['origin_name'] = routes['origin_city'] + ', ' + routes['origin_state']
    routes['dest_name'] = routes['dest_city'] + ', ' + routes['dest_state']
    
    # Calculate route statistics
    stats = []
    for (origin, dest), group in routes.groupby(['origin_name', 'dest_name']):
        flight_count = len(group)
        if flight_count > 100:  # Only include routes with more than 100 flights
            avg_delay = group['delay'].mean()
            total_distance = group['distance'].sum()
            delay_ratio = (group['delay'] > 15).mean()
            
            stats.append({
                'origin_name': origin,
                'dest_name': dest,
                'flight_count': flight_count,
                'avg_delay': round(avg_delay, 2),
                'total_distance': total_distance,
                'delay_ratio': round(delay_ratio, 3)
            })
    
    route_stats = pd.DataFrame(stats)
    return route_stats.sort_values('flight_count', ascending=False)

def create_visualizations(airport_stats, route_stats):
    """Create three different visualizations"""
    # 1. Top 10 Airports by Average Delay
    plt.figure(figsize=(12, 6))
    top_10 = airport_stats.head(10)
    sns.barplot(data=top_10, x='origin', y='avg_delay')
    plt.title('Top 10 Airports by Average Delay')
    plt.xlabel('Airport Code')
    plt.ylabel('Average Delay (minutes)')
    plt.xticks(rotation=45)
    
    # Add airport names as annotations
    for i, row in enumerate(top_10.itertuples()):
        plt.text(i, row.avg_delay, f'{row.avg_delay:.1f}m\n{row.location}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('1_airport_delays.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Delay Ratio vs Flight Volume
    plt.figure(figsize=(10, 6))
    plt.scatter(airport_stats['total_flights'], 
               airport_stats['delay_ratio'],
               alpha=0.5)
    plt.title('Airport Delay Ratio vs Flight Volume')
    plt.xlabel('Total Number of Flights (log scale)')
    plt.ylabel('Delay Ratio (delays > 15 mins)')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for extreme cases
    for _, row in airport_stats.nlargest(3, 'delay_ratio').iterrows():
        plt.annotate(f"{row['origin']}\n{row['location']}", 
                    (row['total_flights'], row['delay_ratio']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig('2_delay_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top Routes Analysis
    plt.figure(figsize=(12, 8))
    top_15 = route_stats.head(15)
    scatter = plt.scatter(top_15['flight_count'], 
                         top_15['avg_delay'],
                         s=top_15['total_distance']/5000,
                         alpha=0.6)
    plt.title('Top 15 Routes: Volume vs Delays')
    plt.xlabel('Number of Flights')
    plt.ylabel('Average Delay (minutes)')
    plt.grid(True, alpha=0.3)
    
    # Add route labels for top 5
    for _, row in top_15.head().iterrows():
        plt.annotate(f"{row['origin_name']} →\n{row['dest_name']}", 
                    (row['flight_count'], row['avg_delay']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8)
    
    # Add legend for bubble size
    legend_elements = [plt.scatter([], [], s=s, 
                                 label=f'{int(s*5000/1000000)}M miles', 
                                 alpha=0.6)
                      for s in [100, 500, 1000]]
    plt.legend(handles=legend_elements, title='Total Distance', 
              labelspacing=2, title_fontsize=10)
    
    plt.tight_layout()
    plt.savefig('3_route_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Load data using pandas
        delays_df, airports_df = load_data()
        
        # Perform analyses
        print("\nAnalyzing airport delays...")
        airport_stats = analyze_airport_delays(delays_df, airports_df)
        
        print("Analyzing route patterns...")
        route_stats = analyze_routes(delays_df, airports_df)
        
        # Create visualizations
        print("Creating visualizations...")
        create_visualizations(airport_stats, route_stats)
        
        # Display results
        print("\nTop 10 Airports with Highest Average Delays:")
        print(airport_stats[['origin', 'location', 'total_flights', 'avg_delay', 'delay_ratio']].head(10))
        
        print("\nTop 10 Busiest Routes:")
        print(route_stats[['origin_name', 'dest_name', 'flight_count', 'avg_delay', 'delay_ratio']].head(10))
        
        # Calculate overall statistics
        total_flights = len(delays_df)
        delayed_flights = (delays_df['delay'] > 15).sum()
        avg_delay = delays_df['delay'].mean()
        
        print(f"\nOverall Statistics:")
        print(f"Total Flights Analyzed: {total_flights:,}")
        print(f"Flights with Significant Delays (>15 mins): {delayed_flights:,}")
        print(f"Average Delay Across All Flights: {avg_delay:.2f} minutes")
        
        print("\nAnalysis Results:")
        print("1. Created airport_delays.png - Shows top 10 airports by average delay")
        print("2. Created delay_ratio.png - Visualizes relationship between flight volume and delay frequency")
        print("3. Created route_analysis.png - Analyzes top 15 routes by volume, delay, and distance")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
