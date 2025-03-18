import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from telethon import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import Channel, Message
from datetime import datetime, timedelta
from collections import Counter
import pytz
import asyncio
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Configure fonts for plots - Changed from Arial to DejaVu Sans for better Unicode support
plt.rcParams['font.family'] = 'DejaVu Sans'  # Better Unicode/emoji support
plt.rcParams['axes.unicode_minus'] = False

# Telegram API credentials
API_ID = ' 1091248 '  # Insert your API ID here
API_HASH = 'f56575fdbd28c9dce37a77eb56e9d3fb'  # Insert your API Hash here
PHONE_NUMBER = '+989150742531'  # Phone number associated with your Telegram account

# Create Telegram client
client = TelegramClient('session_name', API_ID, API_HASH)

async def get_channel_info(channel_username):
    """Retrieve channel information"""
    try:
        entity = await client.get_entity(channel_username)
        channel_full_info = await client(GetFullChannelRequest(channel=entity))
        
        # Basic channel information
        channel_info = {
            'title': entity.title,
            'username': entity.username,
            'id': entity.id,
            'date_created': entity.date,
            'about': channel_full_info.full_chat.about,
            'member_count': channel_full_info.full_chat.participants_count
        }
        
        return entity, channel_info
    except Exception as e:
        print(f"Error getting channel information for {channel_username}: {e}")
        return None, None

async def get_channel_messages(entity, limit=1000):
    """Retrieve channel messages with a specified limit"""
    messages = []
    try:
        async for message in client.iter_messages(entity, limit=limit):
            if message and hasattr(message, 'date'):
                messages.append({
                    'id': message.id,
                    'date': message.date,
                    'views': getattr(message, 'views', 0),
                    'forwards': getattr(message, 'forwards', 0),
                    'text_length': len(message.text) if message.text else 0,
                    'has_media': hasattr(message, 'media') and message.media is not None,
                    'has_document': hasattr(message, 'document') and message.document is not None,
                })
    except Exception as e:
        print(f"Error retrieving messages: {e}")
    
    return messages

def analyze_messages(messages, channel_info):
    """Analyze channel messages"""
    if not messages:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(messages)
    
    # Add time-related columns
    df['hour'] = df['date'].apply(lambda x: x.hour)
    df['day_of_week'] = df['date'].apply(lambda x: x.weekday())
    df['day_name'] = df['date'].apply(lambda x: x.strftime('%A'))
    df['month'] = df['date'].apply(lambda x: x.month)
    df['year'] = df['date'].apply(lambda x: x.year)
    df['date_only'] = df['date'].apply(lambda x: x.date())
    
    # Basic analysis
    analysis = {
        'channel_name': channel_info['title'],
        'channel_username': channel_info['username'],
        'date_created': channel_info['date_created'],
        'total_posts': len(df),
        'first_post_date': df['date'].min() if not df.empty else None,
        'last_post_date': df['date'].max() if not df.empty else None,
        'avg_posts_per_day': len(df) / ((df['date'].max() - df['date'].min()).days + 1) if not df.empty and (df['date'].max() - df['date'].min()).days > 0 else 0,
        'avg_views': df['views'].mean() if 'views' in df.columns else 0,
        'posts_with_media': df['has_media'].sum() if 'has_media' in df.columns else 0,
        'posts_by_hour': df.groupby('hour').size().to_dict(),
        'posts_by_day': df.groupby('day_of_week').size().to_dict(),
        'posts_by_month': df.groupby(['year', 'month']).size().to_dict(),
        'posts_by_date': df.groupby('date_only').size().to_dict(),
    }
    
    return analysis, df

def plot_channel_summary(channel_info, analysis, output_dir):
    """Plot channel summary information"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Display information
    info_text = (
        f"Channel Name: {channel_info['title']}\n"
        f"Username: @{channel_info['username']}\n"
        f"Creation Date: {channel_info['date_created'].strftime('%Y-%m-%d')}\n"
        f"Number of Members: {channel_info['member_count']:,}\n"
        f"Number of Posts: {analysis['total_posts']:,}\n"
        f"First Post: {analysis['first_post_date'].strftime('%Y-%m-%d')}\n"
        f"Last Post: {analysis['last_post_date'].strftime('%Y-%m-%d')}\n"
        f"Average Posts per Day: {analysis['avg_posts_per_day']:.2f}\n"
        f"Percentage of Posts with Media: {(analysis['posts_with_media']/analysis['total_posts']*100):.1f}%"
    )
    
    ax.text(0.5, 0.5, info_text, fontsize=12, ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{channel_info['username']}_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_posts_by_time(analysis, df, output_dir):
    """Plot time-related post distribution charts"""
    channel_username = analysis['channel_username']
    
    # Chart 1: Post distribution by hour
    plt.figure(figsize=(12, 6))
    hours = list(range(24))
    post_counts = [analysis['posts_by_hour'].get(hour, 0) for hour in hours]
    
    sns.barplot(x=hours, y=post_counts)
    plt.title(f"Post Distribution by Hour for Channel {analysis['channel_name']}", fontsize=14)
    plt.xlabel("Hour", fontsize=12)
    plt.ylabel("Number of Posts", fontsize=12)
    plt.xticks(range(0, 24, 2))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{channel_username}_posts_by_hour.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Post distribution by day of week
    plt.figure(figsize=(12, 6))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = [analysis['posts_by_day'].get(day, 0) for day in range(7)]
    
    sns.barplot(x=days, y=day_counts)
    plt.title(f"Post Distribution by Day of Week for Channel {analysis['channel_name']}", fontsize=14)
    plt.xlabel("Day of Week", fontsize=12)
    plt.ylabel("Number of Posts", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{channel_username}_posts_by_day.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Chart 3: Post frequency trend over time
    plt.figure(figsize=(14, 6))
    posts_by_date = df.groupby('date_only').size()
    posts_by_date.index = pd.to_datetime(posts_by_date.index)
    posts_by_date = posts_by_date.sort_index()
    
    # Trend chart with moving average
    ax = posts_by_date.plot(color='skyblue', alpha=0.6, linewidth=1, figsize=(14, 6))
    posts_by_date.rolling(window=7).mean().plot(color='navy', linewidth=2, ax=ax)
    
    plt.title(f"Post Publishing Trend for Channel {analysis['channel_name']}", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Number of Posts", fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend(['Daily Count', 'Weekly Average'])
    
    # Format date on X axis
    if (posts_by_date.index.max() - posts_by_date.index.min()).days > 365:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{channel_username}_post_trend.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Chart 4: Hour-Day heatmap
    plt.figure(figsize=(12, 8))
    hour_day_counts = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    
    sns.heatmap(hour_day_counts, cmap='YlGnBu', annot=False, fmt='d')
    plt.title(f"Posting Heatmap for Channel {analysis['channel_name']} (Hour-Day)", fontsize=14)
    plt.xlabel("Hour of Day", fontsize=12)
    plt.ylabel("Day of Week", fontsize=12)
    plt.yticks(np.arange(7) + 0.5, ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{channel_username}_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_output_directory(base_dir="telegram_analysis"):
    """Create output directory for saving analysis results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_dir}_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir

async def main():
    """Main function of the program"""
    print("Telegram Channel Analysis Program")
    print("=" * 50)
    
    # Get channels from user
    channels_input = input("Enter Telegram channel usernames (separated by commas): ")
    channel_usernames = [username.strip() for username in channels_input.split(',')]
    
    # Limit number of messages to analyze
    try:
        limit = int(input("How many recent messages would you like to analyze? (Default: 1000): ") or "1000")
    except ValueError:
        limit = 1000
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output will be saved in the {output_dir} folder.")
    
    # Start Telegram connection
    await client.start()
    print("Connected to Telegram.")
    
    all_analyses = []
    
    # Analyze each channel
    for channel_username in channel_usernames:
        print(f"\nAnalyzing channel {channel_username}...")
        
        # Get channel information
        entity, channel_info = await get_channel_info(channel_username)
        if not entity or not channel_info:
            print(f"Channel {channel_username} not found or not accessible.")
            continue
        
        print(f"Channel {channel_info['title']} found. Retrieving messages...")
        
        # Get channel messages
        messages = await get_channel_messages(entity, limit=limit)
        print(f"{len(messages)} messages retrieved.")
        
        if not messages:
            print("No messages found for analysis.")
            continue
        
        # Analyze messages
        analysis, messages_df = analyze_messages(messages, channel_info)
        all_analyses.append(analysis)
        
        # Create charts
        print("Creating charts...")
        plot_channel_summary(channel_info, analysis, output_dir)
        plot_posts_by_time(analysis, messages_df, output_dir)
        
        # Save message DataFrame
        messages_df.to_csv(f"{output_dir}/{channel_username}_messages.csv", index=False)
        
        print(f"Analysis of channel {channel_username} completed.")
    
    # Create comparative charts for multiple channels
    if len(all_analyses) > 1:
        print("\nCreating comparative charts...")
        
        # Compare post counts
        plt.figure(figsize=(12, 6))
        channels = [a['channel_name'] for a in all_analyses]
        post_counts = [a['total_posts'] for a in all_analyses]
        
        sns.barplot(x=channels, y=post_counts)
        plt.title("Comparison of Channel Post Counts", fontsize=14)
        plt.xlabel("Channel", fontsize=12)
        plt.ylabel("Number of Posts", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/channel_comparison_posts.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Compare average posts per day
        plt.figure(figsize=(12, 6))
        avg_posts = [a['avg_posts_per_day'] for a in all_analyses]
        
        sns.barplot(x=channels, y=avg_posts)
        plt.title("Comparison of Average Daily Posts by Channel", fontsize=14)
        plt.xlabel("Channel", fontsize=12)
        plt.ylabel("Average Posts per Day", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/channel_comparison_avg_posts.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Close Telegram connection
    await client.disconnect()
    print("\nAnalysis of all channels completed.")
    print(f"All charts and data have been saved in the {output_dir} folder.")

if __name__ == "__main__":
    asyncio.run(main())
