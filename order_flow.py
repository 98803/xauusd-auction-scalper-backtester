"""
Order Flow Analysis
Analyzes Cumulative Volume Delta, Delta per candle, and aggression metrics
"""

import numpy as np
import pandas as pd


class OrderFlowAnalyzer:
    """Analyze order flow and market microstructure"""
    
    def __init__(self, large_print_threshold=500):
        """
        Initialize order flow analyzer
        
        Args:
            large_print_threshold: Volume threshold for "large" prints
        """
        self.large_print_threshold = large_print_threshold
    
    def calculate_delta(self, df, window=None):
        """
        Calculate Delta per candle (buying volume - selling volume proxy)
        
        Args:
            df: DataFrame with volume data
            window: Optional smoothing window
        
        Returns:
            Series of delta values
        """
        # Delta proxy: positive when close > open, negative when close < open
        # Weighted by volume
        delta = np.where(
            df['close'] > df['open'],
            df['volume'] * (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6),
            -df['volume'] * (df['open'] - df['close']) / (df['high'] - df['low'] + 1e-6)
        )
        
        delta_series = pd.Series(delta, index=df.index)
        
        if window:
            delta_series = delta_series.rolling(window).mean()
        
        return delta_series
    
    def calculate_cvd(self, df):
        """
        Calculate Cumulative Volume Delta (running sum of delta)
        
        Args:
            df: DataFrame with price and volume
        
        Returns:
            Series of CVD values
        """
        delta = self.calculate_delta(df)
        cvd = delta.cumsum()
        
        return cvd
    
    def analyze_delta_divergence(self, df, lookback=20):
        """
        Detect divergences between price and delta
        
        Bullish divergence: Price falling but CVD rising (buying strength)
        Bearish divergence: Price rising but CVD falling (selling strength)
        
        Args:
            df: DataFrame with price and volume
            lookback: Lookback window
        
        Returns:
            DataFrame with divergence signals
        """
        cvd = self.calculate_cvd(df)
        price = df['close']
        
        price_change = price.diff(lookback)
        cvd_change = cvd.diff(lookback)
        
        bullish_div = (price_change < 0) & (cvd_change > 0)
        bearish_div = (price_change > 0) & (cvd_change < 0)
        
        return pd.DataFrame({
            'bullish_divergence': bullish_div,
            'bearish_divergence': bearish_div,
            'price_change': price_change,
            'cvd_change': cvd_change
        }, index=df.index)
    
    def detect_absorption(self, df, window=5, threshold=0.3):
        """
        Detect absorption: aggressive orders failing to move price
        
        Args:
            df: DataFrame with price, volume, and delta
            window: Rolling window
            threshold: Delta threshold as % of volume
        
        Returns:
            Series indicating absorption zones
        """
        delta = self.calculate_delta(df)
        price_change = df['close'].diff().abs()
        
        # Absorption: Large delta but small price change
        large_delta = delta.abs() > (df['volume'].rolling(window).mean() * threshold)
        small_move = price_change < (df['close'] * 0.0005)  # Less than 0.05%
        
        absorption = large_delta & small_move
        
        return absorption
    
    def identify_aggressive_prints(self, df, threshold=None):
        """
        Identify large/aggressive market orders
        
        Args:
            df: DataFrame with volume and large_print_volume columns
            threshold: Large print threshold (uses class default if None)
        
        Returns:
            DataFrame with aggressive print analysis
        """
        if threshold is None:
            threshold = self.large_print_threshold
        
        if 'large_print_volume' not in df.columns:
            return pd.DataFrame(index=df.index)
        
        large_prints = df['large_print_volume'] > 0
        
        if 'large_print_side' in df.columns:
            aggressive_buy = (df['large_print_side'] == 1) & large_prints
            aggressive_sell = (df['large_print_side'] == -1) & large_prints
        else:
            # Infer from delta if side not provided
            delta = self.calculate_delta(df)
            aggressive_buy = large_prints & (delta > 0)
            aggressive_sell = large_prints & (delta < 0)
        
        return pd.DataFrame({
            'large_print': large_prints,
            'aggressive_buy': aggressive_buy,
            'aggressive_sell': aggressive_sell,
            'print_volume': df['large_print_volume']
        }, index=df.index)
    
    def measure_aggression_strength(self, df, window=5):
        """
        Measure overall order flow aggression (0-10 scale)
        
        Args:
            df: DataFrame with order flow data
            window: Rolling window
        
        Returns:
            Series with aggression strength (0-10)
        """
        delta = self.calculate_delta(df)
        delta_abs = delta.abs()
        
        # Normalize delta to 0-10 scale
        max_delta = delta_abs.rolling(window).max() + 1e-6
        aggression = (delta_abs / max_delta * 10).fillna(0)
        
        # Cap at 10
        aggression = aggression.clip(0, 10)
        
        return aggression
    
    def analyze_seller_absorption(self, df, window=5):
        """
        Detect bullish absorption: sellers fail, buyers stepping in
        
        Args:
            df: DataFrame with order flow data
        
        Returns:
            Series with absorption strength
        """
        delta = self.calculate_delta(df)
        price = df['close']
        
        # Selling pressure (negative delta)
        selling_pressure = delta < 0
        
        # Price refusing to fall (higher close than previous)
        price_strength = price > price.shift(1)
        
        # Absorption: Selling pressure + price strength
        absorption = selling_pressure & price_strength
        
        return absorption
    
    def analyze_buyer_absorption(self, df, window=5):
        """
        Detect bearish absorption: buyers fail, sellers stepping in
        
        Args:
            df: DataFrame with order flow data
        
        Returns:
            Series with absorption strength
        """
        delta = self.calculate_delta(df)
        price = df['close']
        
        # Buying pressure (positive delta)
        buying_pressure = delta > 0
        
        # Price refusing to rise (lower close than previous)
        price_weakness = price < price.shift(1)
        
        # Absorption: Buying pressure + price weakness
        absorption = buying_pressure & price_weakness
        
        return absorption


class DeltaProfile:
    """Analyze delta distribution across price levels"""
    
    def __init__(self, price_precision=0.5):
        """Initialize delta profile calculator"""
        self.price_precision = price_precision
    
    def calculate_delta_by_price(self, df, window=None):
        """
        Calculate cumulative delta at each price level
        
        Args:
            df: DataFrame with price and volume
            window: Rolling window size
        
        Returns:
            Dictionary mapping price levels to delta values
        """
        if window is None:
            window = len(df)
        
        analyzer = OrderFlowAnalyzer()
        delta = analyzer.calculate_delta(df)
        
        price_levels = {}
        
        for i in range(len(df)):
            start = max(0, i - window + 1)
            price_window = df['close'].iloc[start:i+1]
            delta_window = delta.iloc[start:i+1]
            
            for price, dlt in zip(price_window, delta_window):
                level = round(price / self.price_precision) * self.price_precision
                price_levels[level] = price_levels.get(level, 0) + dlt
        
        return price_levels


if __name__ == "__main__":
    from data_generator import XAUUSDDataGenerator
    
    print("Generating test data...")
    generator = XAUUSDDataGenerator(days=5)
    df = generator.generate()
    
    print("\nAnalyzing order flow...")
    analyzer = OrderFlowAnalyzer()
    
    delta = analyzer.calculate_delta(df)
    cvd = analyzer.calculate_cvd(df)
    
    print(f"Latest Delta: {delta.iloc[-1]:.2f}")
    print(f"Latest CVD: {cvd.iloc[-1]:.2f}")
    
    prints = analyzer.identify_aggressive_prints(df)
    print(f"Large prints in last 10 candles: {prints['large_print'].tail(10).sum()}")
    
    aggression = analyzer.measure_aggression_strength(df)
    print(f"Current aggression: {aggression.iloc[-1]:.2f} / 10")
