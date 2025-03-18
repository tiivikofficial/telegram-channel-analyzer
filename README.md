
# Telegram Channel Analyzer

This program is designed to analyze Telegram channel data. Using the Telegram API, it collects various information from channels and provides useful statistical analyses.

## Features

- Extract basic channel information (creation date, member count, description)
- Analyze posts (total count, first and last post dates, average daily posts)
- Examine time patterns (post distribution by hour and day of week)
- Generate analytical charts (channel summary, post distribution bar charts, activity timeline, heatmaps)
- Compare between channels (when analyzing multiple channels)

## Prerequisites

To use this program, you'll need:

1. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

2. Create an account on [Telegram API](https://my.telegram.org/apps) to obtain API ID and API Hash.

3. Replace the `API_ID`, `API_HASH`, and `PHONE_NUMBER` variables in the code with your information.

## How to Use

1. Run the program:
   ```
   python telegram_analyzer.py
   ```
   
2. Enter the usernames of the channels you want to analyze (separated by commas).

3. Specify the number of recent messages to analyze.

4. The program will automatically save all analyses and charts in a folder named `telegram_analysis_[date]_[time]`.

## Output Examples

The program generates various charts:

- Channel information summary
- Post distribution by hour of day
- Post distribution by day of week
- Post publishing trend over time
- Hour-day heatmap to identify posting patterns

## Technical Features

- Uses the Telethon library to connect to the Telegram API
- Data analysis with Pandas and NumPy
- Visualization with Matplotlib and Seaborn
- Support for different languages and Unicode using the DejaVu Sans font

## Common Issues and Solutions

- If you encounter API access errors, ensure your phone number and API credentials are verified in Telegram.
- To improve performance when retrieving channels with many posts, reduce the message retrieval limit.
- If you see warnings related to Unicode character display, use the default DejaVu Sans font.

## Contributing

Your suggestions and contributions to improve this project are welcome. Please create an Issue first or submit a Pull Request to contribute.

## License

This project is released under the MIT License. See the LICENSE file for more information.
