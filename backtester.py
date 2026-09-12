"""
Complete Auction Market Scalper Backtester
Integrates all components: Market State, Volume Profile, Order Flow, Scoring, Trade Management
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from data_generator import XAUUSDDataGenerator
from market_state import MarketStateAnalyzer
from volume_profile import VolumeProfile
from order_flow import OrderFlowAnalyzer
from scoring_system import ThreePillarScorer, SetupValidator


class Trade:
    """Represents a single trade"""
    
    def __init__(self, trade_id, entry_idx, entry_price, direction, stop, target1, target2=None, runner_qty=0.5):
        self.trade_id = trade_id
        self.entry_idx = entry_idx
        self.entry_price = entry_price
        self.direction = direction  # 'LONG' or 'SHORT'
        self.stop = stop
        self.target1 = target1
        self.target2 = target2
        self.runner_qty = runner_qty
        
        self.exit_idx = None
        self.exit_price = None
        self.exit_reason = None
        
        self.tp1_hit = False
        self.tp2_hit = False
        self.sl_hit = False
        
        self.pnl = 0
        self.pnl_pct = 0
        self.status = 'OPEN'
    
    def calculate_risk(self):
        """Calculate risk in dollars"""
        return abs(self.entry_price - self.stop)
    
    def calculate_rr(self):
        """Calculate risk-to-reward ratio"""
        risk = self.calculate_risk()
        if self.direction == 'LONG':
            reward = self.target1 - self.entry_price
        else:
            reward = self.entry_price - self.target1
        
        if risk == 0:
            return 0
        return reward / risk
    
    def close(self, exit_idx, exit_price, reason):
        """Close the trade"""
        self.exit_idx = exit_idx
        self.exit_price = exit_price
        self.exit_reason = reason
        self.status = 'CLOSED'
        
        if self.direction == 'LONG':
            self.pnl = exit_price - self.entry_price
            self.pnl_pct = (self.pnl / self.entry_price) * 100
        else:
            self.pnl = self.entry_price - exit_price
            self.pnl_pct = (self.pnl / self.entry_price) * 100


class AuctionScalperBacktester:
    """Complete backtester implementing the auction market scalping framework"""
    
    def __init__(self, data, initial_capital=100000, risk_per_trade=0.0025, max_daily_losses=3):
        """
        Initialize backtester
        
        Args:
            data: DataFrame with OHLCV data
            initial_capital: Starting account balance
            risk_per_trade: Risk as % of capital per trade (0.0025 = 0.25%)
            max_daily_losses: Maximum consecutive losses before stopping trading
        """
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_daily_losses = max_daily_losses
        
        self.trades = []
        self.trade_counter = 0
        self.daily_losses = 0
        
        # Initialize analyzers
        self.state_analyzer = MarketStateAnalyzer(lookback=288)
        self.vp_analyzer = VolumeProfile(price_precision=0.5)
        self.of_analyzer = OrderFlowAnalyzer(large_print_threshold=500)
        self.scorer = ThreePillarScorer()
        
        # Pre-calculate metrics
        self._precalculate_metrics()
    
    def _precalculate_metrics(self):
        """Pre-calculate all necessary metrics"""
        print("Calculating Volume Profile...")
        self.poc = self.vp_analyzer.calculate_poc(self.data, window=288)
        self.vah, self.val = self.vp_analyzer.calculate_value_area(self.data, window=288)
        self.hvn, self.lvn = self.vp_analyzer.calculate_hvn_lvn(self.data, window=288)
        
        print("Calculating Order Flow...")
        self.delta = self.of_analyzer.calculate_delta(self.data)
        self.cvd = self.of_analyzer.calculate_cvd(self.data)
        self.aggression = self.of_analyzer.measure_aggression_strength(self.data, window=5)
        self.seller_absorption = self.of_analyzer.analyze_seller_absorption(self.data)
        self.buyer_absorption = self.of_analyzer.analyze_buyer_absorption(self.data)
        
        print("Analyzing Market State...")
        self.market_states = []
        for idx in range(len(self.data)):
            state = self.state_analyzer.classify_market_state(self.data, idx)
            self.market_states.append(state)
        
        print("Metrics pre-calculated successfully!")
    
    def _get_location_info(self, idx):
        """Get location information at index"""
        return {
            'poc': self.poc.iloc[idx] if idx < len(self.poc) else None,
            'vah': self.vah.iloc[idx] if idx < len(self.vah) else None,
            'val': self.val.iloc[idx] if idx < len(self.val) else None,
            'hvn': self.hvn[idx] if idx < len(self.hvn) else [],
            'lvn': self.lvn[idx] if idx < len(self.lvn) else [],
            'tolerance': 0.5
        }
    
    def _get_order_flow_info(self, idx):
        """Get order flow information at index"""
        return {
            'delta': self.delta.iloc[idx] if idx < len(self.delta) else 0,
            'cvd': self.cvd.iloc[idx] if idx < len(self.cvd) else 0,
            'cvd_trend': 1 if (self.cvd.iloc[idx] > self.cvd.iloc[max(0, idx-5)]) else -1,
            'large_prints': 1 if self.data['large_print_volume'].iloc[idx] > 0 else 0,
            'absorption': self.seller_absorption.iloc[idx] or self.buyer_absorption.iloc[idx],
            'follow_through': abs(self.delta.iloc[idx]) > 300
        }
    
    def _evaluate_long_setup(self, idx):
        """Evaluate if conditions support a LONG trade"""
        if idx < 300:
            return None
        
        current_price = self.data['close'].iloc[idx]
        state_info = self.market_states[idx]
        location_info = self._get_location_info(idx)
        order_flow_info = self._get_order_flow_info(idx)
        
        # Score pillars
        state_score = self.scorer.score_market_state(state_info, 'up')
        location_score = self.scorer.score_location(current_price, location_info)
        aggression_score = self.scorer.score_aggression(order_flow_info)
        
        result = self.scorer.score_trade(state_score, location_score, aggression_score)
        
        if result['total_score'] < 5:
            return None
        
        # Find stop (below LVN or support)
        lvn_prices = location_info['lvn']
        if lvn_prices:
            stop_price = min(lvn_prices) - 0.5
        else:
            stop_price = current_price - 1.0
        
        risk = current_price - stop_price
        
        # Calculate position size
        account_risk_dollars = self.capital * self.risk_per_trade
        position_size = account_risk_dollars / risk
        
        # Find targets (at POC, VAH, or swing high)
        poc = location_info['poc']
        target1 = poc if poc and poc > current_price else current_price + (risk * 2)
        target2 = current_price + (risk * 5) if poc else None
        
        return {
            'direction': 'LONG',
            'entry_price': current_price,
            'stop': stop_price,
            'target1': target1,
            'target2': target2,
            'risk': risk,
            'score': result,
            'position_size': position_size
        }
    
    def _evaluate_short_setup(self, idx):
        """Evaluate if conditions support a SHORT trade"""
        if idx < 300:
            return None
        
        current_price = self.data['close'].iloc[idx]
        state_info = self.market_states[idx]
        location_info = self._get_location_info(idx)
        order_flow_info = self._get_order_flow_info(idx)
        
        # Score pillars
        state_score = self.scorer.score_market_state(state_info, 'down')
        location_score = self.scorer.score_location(current_price, location_info)
        aggression_score = self.scorer.score_aggression(order_flow_info)
        
        result = self.scorer.score_trade(state_score, location_score, aggression_score)
        
        if result['total_score'] < 5:
            return None
        
        # Find stop (above HVN or resistance)
        hvn_prices = location_info['hvn']
        if hvn_prices:
            stop_price = max(hvn_prices) + 0.5
        else:
            stop_price = current_price + 1.0
        
        risk = stop_price - current_price
        
        # Calculate position size
        account_risk_dollars = self.capital * self.risk_per_trade
        position_size = account_risk_dollars / risk
        
        # Find targets
        poc = location_info['poc']
        target1 = poc if poc and poc < current_price else current_price - (risk * 2)
        target2 = current_price - (risk * 5) if poc else None
        
        return {
            'direction': 'SHORT',
            'entry_price': current_price,
            'stop': stop_price,
            'target1': target1,
            'target2': target2,
            'risk': risk,
            'score': result,
            'position_size': position_size
        }
    
    def backtest(self):
        """Run the complete backtest"""
        print("\n" + "="*80)
        print("STARTING AUCTION MARKET SCALPER BACKTEST")
        print("="*80)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Risk per Trade: {self.risk_per_trade*100:.2f}%")
        print(f"Data Points: {len(self.data)}")
        print("="*80 + "\n")
        
        for idx in range(300, len(self.data)):
            # Check if we should stop trading (daily loss limit)
            if self.daily_losses >= self.max_daily_losses:
                print(f"[{idx}] Daily loss limit reached. Stopping trading.")
                break
            
            # Update open trades
            self._update_open_trades(idx)
            
            # Evaluate new trade opportunities
            long_setup = self._evaluate_long_setup(idx)
            short_setup = self._evaluate_short_setup(idx)
            
            if long_setup and long_setup['score']['total_score'] >= 6:
                self._enter_trade(idx, long_setup)
            
            if short_setup and short_setup['score']['total_score'] >= 6:
                self._enter_trade(idx, short_setup)
        
        self._finalize_backtest()
    
    def _update_open_trades(self, idx):
        """Update open trades and check exit conditions"""
        current_price = self.data['close'].iloc[idx]
        
        for trade in self.trades:
            if trade.status == 'CLOSED':
                continue
            
            if trade.direction == 'LONG':
                # Check stop loss
                if current_price <= trade.stop:
                    trade.close(idx, trade.stop, 'STOP_LOSS')
                    self.capital -= trade.pnl
                    self.daily_losses += 1
                    print(f"[{idx}] LONG CLOSED (STOP): Entry {trade.entry_price:.2f} → Exit {trade.stop:.2f} | "
                          f"PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
                
                # Check TP1
                elif current_price >= trade.target1 and not trade.tp1_hit:
                    trade.tp1_hit = True
                    partial_pnl = (trade.target1 - trade.entry_price) * (1 - trade.runner_qty)
                    self.capital += partial_pnl
                    print(f"[{idx}] LONG TP1 HIT: {trade.target1:.2f} | Partial PnL: ${partial_pnl:.2f}")
                
                # Check TP2
                elif trade.target2 and current_price >= trade.target2 and not trade.tp2_hit:
                    trade.close(idx, trade.target2, 'TARGET_2')
                    self.capital += trade.pnl
                    self.daily_losses = 0  # Reset loss counter on win
                    print(f"[{idx}] LONG CLOSED (TP2): Entry {trade.entry_price:.2f} → Exit {trade.target2:.2f} | "
                          f"PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
            
            else:  # SHORT
                # Check stop loss
                if current_price >= trade.stop:
                    trade.close(idx, trade.stop, 'STOP_LOSS')
                    self.capital -= trade.pnl
                    self.daily_losses += 1
                    print(f"[{idx}] SHORT CLOSED (STOP): Entry {trade.entry_price:.2f} → Exit {trade.stop:.2f} | "
                          f"PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
                
                # Check TP1
                elif current_price <= trade.target1 and not trade.tp1_hit:
                    trade.tp1_hit = True
                    partial_pnl = (trade.entry_price - trade.target1) * (1 - trade.runner_qty)
                    self.capital += partial_pnl
                    print(f"[{idx}] SHORT TP1 HIT: {trade.target1:.2f} | Partial PnL: ${partial_pnl:.2f}")
                
                # Check TP2
                elif trade.target2 and current_price <= trade.target2 and not trade.tp2_hit:
                    trade.close(idx, trade.target2, 'TARGET_2')
                    self.capital += trade.pnl
                    self.daily_losses = 0
                    print(f"[{idx}] SHORT CLOSED (TP2): Entry {trade.entry_price:.2f} → Exit {trade.target2:.2f} | "
                          f"PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
    
    def _enter_trade(self, idx, setup):
        """Enter a new trade"""
        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            entry_idx=idx,
            entry_price=setup['entry_price'],
            direction=setup['direction'],
            stop=setup['stop'],
            target1=setup['target1'],
            target2=setup['target2']
        )
        
        self.trades.append(trade)
        score = setup['score']
        
        print(f"[{idx}] {setup['direction']} ENTRY: {setup['entry_price']:.2f} | "
              f"Stop: {setup['stop']:.2f} | Target: {setup['target1']:.2f} | "
              f"RR: {trade.calculate_rr():.2f}:1 | Score: {score['total_score']}/7 ({score['quality']})")
    
    def _finalize_backtest(self):
        """Finalize backtest and calculate statistics"""
        closed_trades = [t for t in self.trades if t.status == 'CLOSED']
        
        if not closed_trades:
            print("\nNo closed trades. Backtest complete.")
            return
        
        # Calculate statistics
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        total_pnl = sum([t.pnl for t in closed_trades])
        total_return = (total_pnl / self.initial_capital) * 100
        
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {len(winning_trades)} ({win_rate:.2f}%)")
        print(f"Losing Trades: {len(losing_trades)} ({100-win_rate:.2f}%)")
        print(f"\nAverage Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Profit Factor: {abs(sum([t.pnl for t in winning_trades]) / sum([t.pnl for t in losing_trades])):.2f}" if losing_trades else "Inf")
        print(f"\nTotal PnL: ${total_pnl:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print("="*80 + "\n")
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'final_capital': self.capital
        }
    
    def plot_results(self):
        """Plot backtest results"""
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # Price chart with entries/exits
        ax = axes[0]
        ax.plot(self.data.index, self.data['close'], label='Price', linewidth=2, color='black')
        ax.plot(self.data.index, self.poc, label='POC', linewidth=1, alpha=0.7, color='blue')
        ax.plot(self.data.index, self.vah, label='VAH', linewidth=1, alpha=0.5, color='green')
        ax.plot(self.data.index, self.val, label='VAL', linewidth=1, alpha=0.5, color='red')
        
        # Plot trades
        for trade in self.trades:
            if trade.status == 'CLOSED':
                color = 'green' if trade.pnl > 0 else 'red'
                ax.scatter(trade.entry_idx, trade.entry_price, marker='^', s=100, color=color, zorder=5)
                ax.scatter(trade.exit_idx, trade.exit_price, marker='v', s=100, color=color, zorder=5)
        
        ax.set_title('Price Action with Volume Profile & Trades')
        ax.set_ylabel('Price (USD)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # CVD chart
        ax = axes[1]
        ax.plot(self.data.index, self.cvd, label='CVD', linewidth=1.5, color='purple')
        ax.fill_between(self.data.index, self.cvd, alpha=0.3, color='purple')
        ax.set_title('Cumulative Volume Delta')
        ax.set_ylabel('CVD')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Delta chart
        ax = axes[2]
        colors = ['green' if x > 0 else 'red' for x in self.delta]
        ax.bar(self.data.index, self.delta, color=colors, width=1, alpha=0.6)
        ax.set_title('Delta per Candle')
        ax.set_ylabel('Delta')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Aggression chart
        ax = axes[3]
        ax.plot(self.data.index, self.aggression, label='Aggression', linewidth=1.5, color='orange')
        ax.fill_between(self.data.index, self.aggression, alpha=0.3, color='orange')
        ax.set_title('Order Flow Aggression Strength')
        ax.set_ylabel('Aggression (0-10)')
        ax.set_xlabel('Candle Index')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
        print("Chart saved as 'backtest_results.png'")
        plt.show()


if __name__ == "__main__":
    print("Generating synthetic XAUUSD data...")
    generator = XAUUSDDataGenerator(days=30, base_price=2050.0)
    df = generator.generate()
    
    print(f"Generated {len(df)} candles of 5-minute XAUUSD data\n")
    
    # Run backtest
    backtester = AuctionScalperBacktester(
        data=df,
        initial_capital=100000,
        risk_per_trade=0.0025,
        max_daily_losses=3
    )
    
    backtester.backtest()
    backtester.plot_results()
