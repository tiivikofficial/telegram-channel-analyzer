# Telegram Channel Analyzer
# This program fetches and analyzes Telegram channel data using the Telegram API.

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from telethon import TelegramClient, errors
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import Channel, User
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib font to DejaVu Sans for better Unicode support
plt.rcParams['font.family'] = 'DejaVu Sans'

# Telegram API credentials - DO NOT hardcode these in production!
# Use environment variables or a config file instead
API_ID = ''  # Insert your API ID here
API_HASH = ''  # Insert your API hash here
PHONE_NUMBER = ''  # Insert your phone number here

class TelegramAnalyzer:
    def __init__(self, api_id, api_hash, phone):
        """
        Initialize the Telegram Analyzer with API credentials.
        
        Args:
            api_id (str): Telegram API ID
            api_hash (str): Telegram API Hash
            phone (str): Phone number associated with the Telegram account
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.client = None
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"telegram_analysis_{timestamp}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    async def initialize_client(self):
        """
        Initialize and start the Telegram client.
        """
        self.client = TelegramClient('session_name', self.api_id, self.api_hash)
        await self.client.start(phone=self.phone)
        logger.info("Client initialized successfully")
        
    async def get_channel_info(self, channel_username):
        """
        Get basic information about the channel.
        
        Args:
            channel_username (str): Username of the Telegram channel
            
        Returns:
            dict: Dictionary containing channel information
        """
        try:
            # Get the channel entity
            channel_entity = await self.client.get_entity(channel_username)
            
            # Get full channel information
            full_channel = await self.client(GetFullChannelRequest(channel=channel_entity))
            
            # IMPORTANT: Fix for the channel creation date issue
            # Properly extract the creation date from the channel entity
            creation_date = channel_entity.date
            
            # Gather channel information
            channel_info = {
                "username": channel_username,
                "title": channel_entity.title,
                "id": channel_entity.id,
                "creation_date": creation_date.strftime("%B %d, %Y"),  # Format as Month Day, Year
                "about": full_channel.full_chat.about,
                "member_count": full_channel.full_chat.participants_count,
                "photo": channel_entity.photo is not None,
            }
            
            logger.info(f"Retrieved information for channel: {channel_username}")
            return channel_info
            
        except errors.FloodWaitError as e:
            logger.error(f"Rate limit exceeded. Wait for {e.seconds} seconds")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to get channel info: {e}")
            return None
            
    async def get_channel_messages(self, channel_username, limit=100):
        """
        Fetch messages from the channel.
        
        Args:
            channel_username (str): Username of the Telegram channel
            limit (int): Maximum number of messages to retrieve
            
        Returns:
            list: List of message dictionaries
        """
        messages = []
        try:
            # Get the channel entity
            channel_entity = await self.client.get_entity(channel_username)
            
            # Fetch messages
            async for message in self.client.iter_messages(channel_entity, limit=limit):
                if message.message:  # Skip empty messages
                    message_data = {
                        "id": message.id,
                        "date": message.date,
                        "text": message.message,
                        "views": getattr(message, 'views', 0),
                        "forwards": getattr(message, 'forwards', 0),
                        "has_media": message.media is not None,
                        "has_document": hasattr(message, 'document') and message.document is not None,
                        "has_photo": hasattr(message, 'photo') and message.photo is not None,
                    }
                    messages.append(message_data)
            
            logger.info(f"Retrieved {len(messages)} messages from channel: {channel_username}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get channel messages: {e}")
            return []
            
    def analyze_messages(self, messages):
        """
        Analyze channel messages to extract insights.
        
        Args:
            messages (list): List of message dictionaries
            
        Returns:
            dict: Dictionary containing analysis results
        """
        if not messages:
            logger.warning("No messages to analyze")
            return {}
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(messages)
        
        # Sort by date
        df = df.sort_values('date')
        
        # Extract features
        first_post_date = df['date'].min()
        last_post_date = df['date'].max()
        total_days = (last_post_date - first_post_date).days + 1
        
        # Add hour and day columns for time analysis
        df['hour'] = df['date'].apply(lambda x: x.hour)
        df['day_of_week'] = df['date'].apply(lambda x: x.strftime('%A'))
        df['date_only'] = df['date'].apply(lambda x: x.date())
        
        # Count posts per day
        posts_per_day = df.groupby('date_only').size()
        
        # Create day of week order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Analysis results
        analysis = {
            "total_posts": len(df),
            "first_post_date": first_post_date.strftime("%B %d, %Y"),
            "last_post_date": last_post_date.strftime("%B %d, %Y"),
            "total_days": total_days,
            "avg_posts_per_day": len(df) / total_days if total_days > 0 else 0,
            "posts_with_media": df['has_media'].sum(),
            "posts_with_photo": df['has_photo'].sum(),
            "posts_with_document": df['has_document'].sum(),
            "hour_distribution": df.groupby('hour').size().to_dict(),
            "day_distribution": df.groupby('day_of_week').size().to_dict(),
            "day_order": day_order,
            "posts_per_day": posts_per_day.to_dict(),
            "avg_views": df['views'].mean(),
            "max_views": df['views'].max(),
        }
        
        return analysis, df
        
    def generate_charts(self, channel_info, analysis, df, output_prefix):
        """
        Generate visualization charts based on analysis.
        
        Args:
            channel_info (dict): Channel information
            analysis (dict): Analysis results
            df (DataFrame): DataFrame containing message data
            output_prefix (str): Prefix for output filenames
        """
        # Set style
        sns.set(style="whitegrid")
        
        # 1. Channel Summary Chart
        plt.figure(figsize=(10, 6))
        plt.title(f"Channel Summary: {channel_info['title']}")
        
        summary_text = (
            f"Channel Name: {channel_info['title']}\n"
            f"Creation Date: {channel_info['creation_date']}\n"
            f"Member Count: {channel_info['member_count']}\n"
            f"Total Posts: {analysis['total_posts']}\n"
            f"First Post: {analysis['first_post_date']}\n"
            f"Last Post: {analysis['last_post_date']}\n"
            f"Average Posts Per Day: {analysis['avg_posts_per_day']:.2f}\n"
            f"Posts with Media: {analysis['posts_with_media']} ({analysis['posts_with_media']/analysis['total_posts']*100:.1f}%)\n"
            f"Average Views: {analysis['avg_views']:.1f}\n"
            f"Max Views: {analysis['max_views']}\n"
        )
        
        # Remove axes
        plt.axis('off')
        plt.text(0.1, 0.5, summary_text, fontsize=12, va='center')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{output_prefix}_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Hourly Distribution Chart
        plt.figure(figsize=(12, 6))
        
        hours = list(range(24))
        counts = [analysis['hour_distribution'].get(hour, 0) for hour in hours]
        
        sns.barplot(x=hours, y=counts)
        plt.title(f"Post Distribution by Hour of Day - {channel_info['title']}")
        plt.xlabel("Hour of Day (24-hour format)")
        plt.ylabel("Number of Posts")
        plt.xticks(range(0, 24, 2))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{output_prefix}_hourly_distribution.png", dpi=300)
        plt.close()
        
        # 3. Daily Distribution Chart
        plt.figure(figsize=(12, 6))
        
        day_counts = [analysis['day_distribution'].get(day, 0) for day in analysis['day_order']]
        
        sns.barplot(x=analysis['day_order'], y=day_counts)
        plt.title(f"Post Distribution by Day of Week - {channel_info['title']}")
        plt.xlabel("Day of Week")
        plt.ylabel("Number of Posts")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{output_prefix}_daily_distribution.png", dpi=300)
        plt.close()
        
        # 4. Time Series Analysis
        if len(analysis['posts_per_day']) > 1:
            plt.figure(figsize=(14, 6))
            
            dates = list(analysis['posts_per_day'].keys())
            counts = list(analysis['posts_per_day'].values())
            
            # Create a date range for all days
            all_dates = pd.date_range(min(dates), max(dates))
            
            # Create a complete series with zeros for missing dates
            complete_series = pd.Series(
                [analysis['posts_per_day'].get(date.date(), 0) for date in all_dates],
                index=all_dates
            )
            
            # Plot the time series
            plt.plot(complete_series.index, complete_series.values, marker='o', linestyle='-', alpha=0.7)
            plt.title(f"Posting Activity Over Time - {channel_info['title']}")
            plt.xlabel("Date")
            plt.ylabel("Number of Posts")
            plt.grid(True)
            
            # Format x-axis to show date labels nicely
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{output_prefix}_time_series.png", dpi=300)
            plt.close()
            
        # 5. Heatmap: Hour vs Day of Week
        plt.figure(figsize=(12, 8))
        
        # Create day-hour pivot table
        day_hour_counts = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        pivot_data = day_hour_counts.pivot(index='day_of_week', columns='hour', values='count')
        
        # Reindex to ensure all days are in correct order
        pivot_data = pivot_data.reindex(analysis['day_order'])
        
        # Fill NaN values with 0
        pivot_data = pivot_data.fillna(0)
        
        # Create the heatmap
        sns.heatmap(pivot_data, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=.5)
        plt.title(f"Activity Heatmap: Day of Week vs Hour - {channel_info['title']}")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{output_prefix}_heatmap.png", dpi=300)
        plt.close()
        
    def save_analysis_to_file(self, channel_info, analysis, output_prefix):
        """
        Save analysis results to a text file.
        
        Args:
            channel_info (dict): Channel information
            analysis (dict): Analysis results
            output_prefix (str): Prefix for output filename
        """
        with open(f"{self.output_dir}/{output_prefix}_analysis.txt", 'w', encoding='utf-8') as f:
            f.write(f"CHANNEL ANALYSIS: {channel_info['title']}\n")
            f.write("="*50 + "\n\n")
            
            f.write("CHANNEL INFORMATION:\n")
            f.write(f"Username: @{channel_info['username']}\n")
            f.write(f"Title: {channel_info['title']}\n")
            f.write(f"ID: {channel_info['id']}\n")
            f.write(f"Creation Date: {channel_info['creation_date']}\n")
            f.write(f"Member Count: {channel_info['member_count']}\n")
            f.write(f"About: {channel_info['about']}\n\n")
            
            f.write("POST ANALYSIS:\n")
            f.write(f"Total Posts Analyzed: {analysis['total_posts']}\n")
            f.write(f"First Post Date: {analysis['first_post_date']}\n")
            f.write(f"Last Post Date: {analysis['last_post_date']}\n")
            f.write(f"Total Days: {analysis['total_days']}\n")
            f.write(f"Average Posts Per Day: {analysis['avg_posts_per_day']:.2f}\n")
            f.write(f"Posts with Media: {analysis['posts_with_media']} ({analysis['posts_with_media']/analysis['total_posts']*100:.1f}%)\n")
            f.write(f"Posts with Photos: {analysis['posts_with_photo']} ({analysis['posts_with_photo']/analysis['total_posts']*100:.1f}%)\n")
            f.write(f"Posts with Documents: {analysis['posts_with_document']} ({analysis['posts_with_document']/analysis['total_posts']*100:.1f}%)\n")
            f.write(f"Average Views: {analysis['avg_views']:.1f}\n")
            f.write(f"Maximum Views: {analysis['max_views']}\n\n")
            
            f.write("HOURLY DISTRIBUTION:\n")
            for hour in range(24):
                count = analysis['hour_distribution'].get(hour, 0)
                percentage = count / analysis['total_posts'] * 100 if analysis['total_posts'] > 0 else 0
                f.write(f"{hour:02d}:00 - {hour:02d}:59: {count} posts ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("DAILY DISTRIBUTION:\n")
            for day in analysis['day_order']:
                count = analysis['day_distribution'].get(day, 0)
                percentage = count / analysis['total_posts'] * 100 if analysis['total_posts'] > 0 else 0
                f.write(f"{day}: {count} posts ({percentage:.1f}%)\n")
            
            logger.info(f"Analysis saved to {self.output_dir}/{output_prefix}_analysis.txt")
            
    async def analyze_channel(self, channel_username, limit=1000):
        """
        Perform complete analysis for a single channel.
        
        Args:
            channel_username (str): Username of the Telegram channel
            limit (int): Maximum number of messages to retrieve
            
        Returns:
            tuple: (channel_info, analysis, dataframe)
        """
        # Remove @ if present in the channel username
        if channel_username.startswith('@'):
            channel_username = channel_username[1:]
            
        # Get channel information
        channel_info = await self.get_channel_info(channel_username)
        
        if not channel_info:
            logger.error(f"Failed to analyze channel {channel_username}")
            return None, None, None
            
        # Get channel messages
        messages = await self.get_channel_messages(channel_username, limit)
        
        # Analyze messages
        analysis, df = self.analyze_messages(messages)
        
        # Generate output filename prefix
        output_prefix = channel_username
        
        # Save analysis results
        self.save_analysis_to_file(channel_info, analysis, output_prefix)
        
        # Generate charts
        self.generate_charts(channel_info, analysis, df, output_prefix)
        
        logger.info(f"Analysis completed for channel: {channel_username}")
        return channel_info, analysis, df
        
    async def compare_channels(self, channel_data_list):
        """
        Generate comparison charts for multiple channels.
        
        Args:
            channel_data_list: List of (channel_info, analysis, df) tuples
        """
        if len(channel_data_list) < 2:
            logger.info("Not enough channels to compare")
            return
            
        # 1. Posts per day comparison
        plt.figure(figsize=(12, 6))
        
        channels = []
        posts_per_day = []
        
        for channel_info, analysis, _ in channel_data_list:
            if channel_info and analysis:
                channels.append(channel_info['title'])
                posts_per_day.append(analysis['avg_posts_per_day'])
                
        sns.barplot(x=channels, y=posts_per_day)
        plt.title("Average Posts Per Day Comparison")
        plt.xlabel("Channel")
        plt.ylabel("Average Posts Per Day")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/comparison_posts_per_day.png", dpi=300)
        plt.close()
        
        # 2. Member count comparison
        plt.figure(figsize=(12, 6))
        
        channels = []
        member_counts = []
        
        for channel_info, _, _ in channel_data_list:
            if channel_info:
                channels.append(channel_info['title'])
                member_counts.append(channel_info['member_count'])
                
        sns.barplot(x=channels, y=member_counts)
        plt.title("Member Count Comparison")
        plt.xlabel("Channel")
        plt.ylabel("Number of Members")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/comparison_member_count.png", dpi=300)
        plt.close()
        
        # 3. Activity patterns comparison
        plt.figure(figsize=(14, 8))
        
        for idx, (channel_info, analysis, _) in enumerate(channel_data_list):
            if channel_info and analysis:
                hours = list(range(24))
                counts = [analysis['hour_distribution'].get(hour, 0) for hour in hours]
                
                # Convert to percentages for fair comparison
                total = sum(counts)
                if total > 0:
                    percentages = [count / total * 100 for count in counts]
                else:
                    percentages = [0] * 24
                    
                plt.plot(hours, percentages, marker='o', label=channel_info['title'])
                
        plt.title("Hourly Activity Pattern Comparison (Percentage)")
        plt.xlabel("Hour of Day")
        plt.ylabel("Percentage of Posts")
        plt.xticks(range(0, 24, 2))
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/comparison_hourly_pattern.png", dpi=300)
        plt.close()
        
        # Save comparison summary to file
        with open(f"{self.output_dir}/channel_comparison.txt", 'w', encoding='utf-8') as f:
            f.write("CHANNEL COMPARISON\n")
            f.write("="*50 + "\n\n")
            
            for channel_info, analysis, _ in channel_data_list:
                if channel_info and analysis:
                    f.write(f"Channel: {channel_info['title']}\n")
                    f.write(f"  Creation Date: {channel_info['creation_date']}\n")
                    f.write(f"  Member Count: {channel_info['member_count']}\n")
                    f.write(f"  Total Posts: {analysis['total_posts']}\n")
                    f.write(f"  Average Posts Per Day: {analysis['avg_posts_per_day']:.2f}\n")
                    f.write(f"  Average Views: {analysis['avg_views']:.1f}\n\n")
                    
            logger.info(f"Comparison summary saved to {self.output_dir}/channel_comparison.txt")

async def main():
    """
    Main function to run the Telegram channel analyzer.
    """
    # Check if API credentials are provided
    if not API_ID or not API_HASH or not PHONE_NUMBER:
        print("Please set your API_ID, API_HASH, and PHONE_NUMBER in the script.")
        print("You can obtain these from https://my.telegram.org/apps")
        return
        
    # Create analyzer instance
    analyzer = TelegramAnalyzer(API_ID, API_HASH, PHONE_NUMBER)
    
    # Initialize client
    await analyzer.initialize_client()
    
    # Get channel usernames from user
    channel_input = input("Enter channel usernames (separated by comma): ")
    channels = [channel.strip() for channel in channel_input.split(',')]
    
    # Get message limit from user
    try:
        limit = int(input("Enter number of recent messages to analyze (default: 1000): ") or 1000)
    except ValueError:
        limit = 1000
        
    print(f"\nAnalyzing {len(channels)} channels, retrieving up to {limit} messages each...")
    print(f"Results will be saved to: {analyzer.output_dir}\n")
    
    # Analyze each channel
    channel_data_list = []
    for channel in channels:
        print(f"Analyzing channel: {channel}")
        channel_info, analysis, df = await analyzer.analyze_channel(channel, limit)
        if channel_info and analysis:
            channel_data_list.append((channel_info, analysis, df))
            print(f"✓ Completed analysis for {channel}")
        else:
            print(f"✗ Failed to analyze {channel}")
    
    # Compare channels if multiple were analyzed
    if len(channel_data_list) > 1:
        print("\nGenerating channel comparisons...")
        await analyzer.compare_channels(channel_data_list)
        print("✓ Comparison completed")
        
    # Close the client
    await analyzer.client.disconnect()
    
    print(f"\nAnalysis complete! All results saved to: {analyzer.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
