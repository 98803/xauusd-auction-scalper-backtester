"""
Volume Profile Analysis
Calculates Point of Control (POC), Value Area High/Low, and Low Volume Nodes
"""

import numpy as np
import pandas as pd


class VolumeProfile:
    """Calculate volume profile metrics for auction analysis"""
    
    def __init__(self, price_precision=0.5):
        """
        Initialize volume profile calculator
        
        Args:
            price_precision: Price level granularity (e.g., 0.5 = $0.50 levels for gold)
        """
        self.price_precision = price_precision
    
    def calculate_poc(self, df, window=None):
        """
        Calculate Point of Control (price with most volume)
        
        Args:
            df: DataFrame with close and volume
            window: Rolling window size. If None, uses entire dataset
        
        Returns:
            Series of POC values
        """
        if window is None:
            window = len(df)
        
        poc_values = []
        
        for i in range(len(df)):
            start = max(0, i - window + 1)
            price_range = df['close'].iloc[start:i+1]
            volume_range = df['volume'].iloc[start:i+1]
            
            # Round prices to precision level
            price_levels = np.round(price_range / self.price_precision) * self.price_precision
            
            # Group volume by price level
            level_volumes = {}
            for price, vol in zip(price_levels, volume_range):
                level_volumes[price] = level_volumes.get(price, 0) + vol
            
            # POC is the price level with max volume
            poc = max(level_volumes, key=level_volumes.get)
            poc_values.append(poc)
        
        return pd.Series(poc_values, index=df.index)
    
    def calculate_value_area(self, df, window=None, percentile=0.70):
        """
        Calculate Value Area High and Low (price range containing X% of volume)
        
        Args:
            df: DataFrame with close and volume
            window: Rolling window size
            percentile: Percentage of volume to include (default 70%)
        
        Returns:
            Tuple of (VAH series, VAL series)
        """
        if window is None:
            window = len(df)
        
        vah_values = []
        val_values = []
        
        for i in range(len(df)):
            start = max(0, i - window + 1)
            price_range = df['close'].iloc[start:i+1]
            volume_range = df['volume'].iloc[start:i+1]
            
            # Round prices to precision level
            price_levels = np.round(price_range / self.price_precision) * self.price_precision
            
            # Group volume by price level
            level_volumes = {}
            for price, vol in zip(price_levels, volume_range):
                level_volumes[price] = level_volumes.get(price, 0) + vol
            
            # Sort by price
            sorted_levels = sorted(level_volumes.items(), key=lambda x: x[0])
            
            # Find price range containing percentile% of volume
            total_vol = sum([v for _, v in sorted_levels])
            target_vol = total_vol * percentile
            
            cumulative_vol = 0
            value_levels = []
            
            for price, vol in sorted_levels:
                if cumulative_vol < target_vol:
                    value_levels.append(price)
                    cumulative_vol += vol
            
            if value_levels:
                vah = max(value_levels)
                val = min(value_levels)
            else:
                vah = price_range.max()
                val = price_range.min()
            
            vah_values.append(vah)
            val_values.append(val)
        
        return pd.Series(vah_values, index=df.index), pd.Series(val_values, index=df.index)
    
    def calculate_hvn_lvn(self, df, window=None, num_levels=5):
        """
        Calculate High Volume Nodes (HVN) and Low Volume Nodes (LVN)
        
        Args:
            df: DataFrame with close and volume
            window: Rolling window size
            num_levels: Number of price levels to consider
        
        Returns:
            Tuple of (HVN levels, LVN levels)
        """
        if window is None:
            window = len(df)
        
        hvn_list = []
        lvn_list = []
        
        for i in range(len(df)):
            start = max(0, i - window + 1)
            price_range = df['close'].iloc[start:i+1]
            volume_range = df['volume'].iloc[start:i+1]
            
            # Round prices to precision level
            price_levels = np.round(price_range / self.price_precision) * self.price_precision
            
            # Group volume by price level
            level_volumes = {}
            for price, vol in zip(price_levels, volume_range):
                level_volumes[price] = level_volumes.get(price, 0) + vol
            
            if not level_volumes:
                hvn_list.append([])
                lvn_list.append([])
                continue
            
            # Sort by volume
            sorted_by_vol = sorted(level_volumes.items(), key=lambda x: x[1], reverse=True)
            
            # Top levels = HVN, Bottom levels = LVN
            hvn = [price for price, _ in sorted_by_vol[:num_levels]]
            lvn = [price for price, _ in sorted_by_vol[-num_levels:]]
            
            hvn_list.append(hvn)
            lvn_list.append(lvn)
        
        return hvn_list, lvn_list
    
    def identify_imbalances(self, df, window=None, threshold=0.3):
        """
        Identify price levels with volume imbalances
        
        Args:
            df: DataFrame with close and volume
            window: Rolling window size
            threshold: Imbalance ratio threshold
        
        Returns:
            DataFrame with imbalance levels
        """
        if window is None:
            window = len(df)
        
        imbalances = []
        
        for i in range(len(df)):
            start = max(0, i - window + 1)
            high_range = df['high'].iloc[start:i+1]
            low_range = df['low'].iloc[start:i+1]
            volume_range = df['volume'].iloc[start:i+1]
            
            # Identify gaps or low-volume areas
            for j in range(start + 1, i + 1):
                prev_high = high_range.iloc[j - start - 1]
                curr_low = low_range.iloc[j - start]
                
                if curr_low > prev_high:  # Gap up
                    gap_size = curr_low - prev_high
                    imbalances.append({
                        'index': i,
                        'type': 'gap_up',
                        'level': prev_high + gap_size / 2,
                        'size': gap_size
                    })
                
                curr_high = high_range.iloc[j - start]
                prev_low = low_range.iloc[j - start - 1]
                
                if prev_low > curr_high:  # Gap down
                    gap_size = prev_low - curr_high
                    imbalances.append({
                        'index': i,
                        'type': 'gap_down',
                        'level': prev_low - gap_size / 2,
                        'size': gap_size
                    })
        
        return pd.DataFrame(imbalances) if imbalances else pd.DataFrame()


class SessionProfile:
    """Analyze session-specific volume profiles"""
    
    def __init__(self, df, session_type='ny'):
        """
        Initialize session profile
        
        Args:
            df: DataFrame with timestamp index
            session_type: 'london', 'ny', 'asia', or 'all'
        """
        self.df = df
        self.session_type = session_type
        self._filter_session()
    
    def _filter_session(self):
        """Filter data by session"""
        hour = self.df.index.hour
        
        if self.session_type == 'asia':
            mask = (hour >= 0) & (hour < 8)
        elif self.session_type == 'london':
            mask = (hour >= 8) & (hour < 16)
        elif self.session_type == 'ny':
            mask = (hour >= 16) | (hour < 24)
        else:
            mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        self.session_data = self.df[mask]
    
    def get_profile(self):
        """Get volume profile for the session"""
        vp = VolumeProfile()
        poc = vp.calculate_poc(self.session_data, window=len(self.session_data))
        vah, val = vp.calculate_value_area(self.session_data, window=len(self.session_data))
        hvn, lvn = vp.calculate_hvn_lvn(self.session_data, window=len(self.session_data))
        
        return {
            'poc': poc.iloc[-1] if len(poc) > 0 else None,
            'vah': vah.iloc[-1] if len(vah) > 0 else None,
            'val': val.iloc[-1] if len(val) > 0 else None,
            'hvn': hvn[-1] if hvn else [],
            'lvn': lvn[-1] if lvn else [],
        }


if __name__ == "__main__":
    from data_generator import XAUUSDDataGenerator
    
    print("Generating test data...")
    generator = XAUUSDDataGenerator(days=5)
    df = generator.generate()
    
    print("\nCalculating volume profile...")
    vp = VolumeProfile()
    
    poc = vp.calculate_poc(df, window=288)
    vah, val = vp.calculate_value_area(df, window=288)
    hvn, lvn = vp.calculate_hvn_lvn(df, window=288)
    
    print(f"Latest POC: {poc.iloc[-1]:.2f}")
    print(f"Latest VAH: {vah.iloc[-1]:.2f}")
    print(f"Latest VAL: {val.iloc[-1]:.2f}")
    print(f"Latest HVN: {hvn[-1]}")
    print(f"Latest LVN: {lvn[-1]}")
