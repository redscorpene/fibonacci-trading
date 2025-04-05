# convert_mt5_data.py
import pandas as pd
import sys

def convert_mt5_data(input_file, output_file):
    print(f"Reading data from {input_file}...")
    
    # Create lists to hold the data
    times = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    try:
        # Read the file line by line
        with open(input_file, 'r') as file:
            lines = file.readlines()
        
        # Process each line
        for line in lines:
            # Remove quotes
            line = line.replace('"', '')
            parts = line.strip().split()
            
            # Skip header line
            if len(parts) >= 9 and not (parts[0] == '<DATE>' or parts[0] == 'DATE'):
                # Extract date and time
                date = parts[0]
                time = parts[1]
                timestamp = f"{date} {time}"
                
                # Extract price data
                try:
                    open_price = float(parts[2])
                    high_price = float(parts[3])
                    low_price = float(parts[4])
                    close_price = float(parts[5])
                    volume = float(parts[6])
                    
                    # Append to lists
                    times.append(timestamp)
                    opens.append(open_price)
                    highs.append(high_price)
                    lows.append(low_price)
                    closes.append(close_price)
                    volumes.append(volume)
                except (ValueError, IndexError) as e:
                    print(f"Error processing line: {line}")
                    print(f"Error details: {e}")
                    continue
    
        # Create DataFrame
        df = pd.DataFrame({
            'time': pd.to_datetime(times),
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        # Save formatted data
        df.to_csv(output_file, index=False)
        print(f"Converted {len(df)} candles to the proper format.")
        print(f"Saved to {output_file}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        return True
    
    except Exception as e:
        print(f"Error converting data: {e}")
        return False

if __name__ == "__main__":
    # Default file paths
    input_file = 'data/btcusd_raw.csv'
    output_file = 'data/btcusd_formatted.csv'
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    convert_mt5_data(input_file, output_file)