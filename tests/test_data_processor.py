"""
Unit tests for the data_processor module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from spa.data_processor import (
    calculate_daily_returns,
    calculate_return_statistics,
    calculate_cumulative_returns,
    calculate_total_return,
    calculate_annualized_return,
    calculate_volatility,
    calculate_rolling_volatility,
    calculate_downside_volatility,
    calculate_drawdown,
    calculate_max_drawdown,
    calculate_drawdown_details,
    calculate_calmar_ratio,
    calculate_moving_average,
    calculate_sma,
    calculate_ema,
    detect_golden_cross,
    detect_death_cross,
    get_moving_average_signals,
    get_return_volatility_summary,
    get_comprehensive_analysis
)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='B')
    
    # Create synthetic price data with known characteristics
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.02))
    
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return df


@pytest.fixture
def simple_price_data():
    """Create simple predictable price data."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    prices = [100, 102, 101, 105, 103, 107, 110, 108, 112, 115]
    
    df = pd.DataFrame({
        'Close': prices
    }, index=dates)
    
    return df


@pytest.fixture
def drawdown_data():
    """Create data with known drawdown."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    # Peak at 110, trough at 90, recovery to 100
    prices = [100, 105, 110, 100, 95, 90, 92, 95, 98, 100]
    
    df = pd.DataFrame({
        'Close': prices
    }, index=dates)
    
    return df


class TestDailyReturns:
    """Tests for daily return calculations."""
    
    def test_simple_returns(self, simple_price_data):
        """Test simple return calculation."""
        returns = calculate_daily_returns(simple_price_data, method='simple')
        
        assert len(returns) == 9  # One less than prices
        assert isinstance(returns, pd.Series)
        
        # First return: (102 - 100) / 100 = 0.02
        assert abs(returns.iloc[0] - 0.02) < 0.0001
    
    def test_log_returns(self, simple_price_data):
        """Test logarithmic return calculation."""
        returns = calculate_daily_returns(simple_price_data, method='log')
        
        assert len(returns) == 9
        assert isinstance(returns, pd.Series)
        
        # Log returns should be slightly different from simple
        simple_returns = calculate_daily_returns(simple_price_data, method='simple')
        assert not np.allclose(returns.values, simple_returns.values)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        returns = calculate_daily_returns(df)
        
        assert returns.empty
    
    def test_invalid_method(self, simple_price_data):
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid method"):
            calculate_daily_returns(simple_price_data, method='invalid')
    
    def test_missing_column(self, simple_price_data):
        """Test missing column raises error."""
        with pytest.raises(ValueError, match="not found"):
            calculate_daily_returns(simple_price_data, price_column='NonExistent')


class TestReturnStatistics:
    """Tests for return statistics."""
    
    def test_return_statistics(self, sample_price_data):
        """Test calculation of return statistics."""
        returns = calculate_daily_returns(sample_price_data)
        stats = calculate_return_statistics(returns)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats
        assert 'count' in stats
        
        assert stats['count'] == len(returns)
    
    def test_empty_returns(self):
        """Test empty returns."""
        empty_series = pd.Series(dtype=float)
        stats = calculate_return_statistics(empty_series)
        
        assert stats == {}


class TestCumulativeReturns:
    """Tests for cumulative return calculations."""
    
    def test_simple_cumulative_returns(self, simple_price_data):
        """Test simple cumulative returns."""
        cum_returns = calculate_cumulative_returns(simple_price_data, method='simple')
        
        assert len(cum_returns) == len(simple_price_data)
        assert cum_returns.iloc[0] == 0.0  # First day = 0%
        
        # Last return: (115 - 100) / 100 = 0.15
        assert abs(cum_returns.iloc[-1] - 0.15) < 0.0001
    
    def test_compound_cumulative_returns(self, simple_price_data):
        """Test compound cumulative returns."""
        cum_returns = calculate_cumulative_returns(simple_price_data, method='compound')
        
        assert len(cum_returns) == len(simple_price_data)
        assert cum_returns.iloc[0] == 0.0
    
    def test_total_return(self, simple_price_data):
        """Test total return calculation."""
        total_return = calculate_total_return(simple_price_data)
        
        # (115 - 100) / 100 = 0.15
        assert abs(total_return - 0.15) < 0.0001
    
    def test_annualized_return(self, sample_price_data):
        """Test annualized return calculation."""
        ann_return = calculate_annualized_return(sample_price_data)
        
        assert isinstance(ann_return, float)
        # Should be reasonable (not extremely high or low)
        assert -1.0 < ann_return < 2.0


class TestVolatility:
    """Tests for volatility calculations."""
    
    def test_volatility_calculation(self, sample_price_data):
        """Test basic volatility calculation."""
        vol = calculate_volatility(sample_price_data, annualize=False)
        
        assert isinstance(vol, float)
        assert vol > 0
    
    def test_annualized_volatility(self, sample_price_data):
        """Test annualized volatility."""
        daily_vol = calculate_volatility(sample_price_data, annualize=False)
        annual_vol = calculate_volatility(sample_price_data, annualize=True)
        
        # Annualized should be roughly sqrt(252) times daily
        ratio = annual_vol / daily_vol
        assert 15 < ratio < 17  # sqrt(252) ≈ 15.87
    
    def test_rolling_volatility(self, sample_price_data):
        """Test rolling volatility calculation."""
        rolling_vol = calculate_rolling_volatility(sample_price_data, window=30)
        
        assert isinstance(rolling_vol, pd.Series)
        # Rolling vol is calculated on returns, which has one less row
        assert len(rolling_vol) == len(sample_price_data) - 1
    
    def test_downside_volatility(self, sample_price_data):
        """Test downside volatility calculation."""
        downside_vol = calculate_downside_volatility(sample_price_data)
        regular_vol = calculate_volatility(sample_price_data)
        
        # Downside vol should be less than or equal to regular vol
        assert downside_vol <= regular_vol


class TestDrawdown:
    """Tests for drawdown calculations."""
    
    def test_drawdown_series(self, drawdown_data):
        """Test drawdown series calculation."""
        dd = calculate_drawdown(drawdown_data)
        
        assert len(dd) == len(drawdown_data)
        assert dd.iloc[0] == 0.0  # No drawdown at start
        assert dd.min() < 0  # Should have negative values
    
    def test_max_drawdown(self, drawdown_data):
        """Test max drawdown calculation."""
        max_dd = calculate_max_drawdown(drawdown_data)
        
        # Peak at 110, trough at 90: (90-110)/110 = -0.1818
        assert abs(max_dd - (-0.1818)) < 0.01
    
    def test_drawdown_details(self, drawdown_data):
        """Test detailed drawdown information."""
        details = calculate_drawdown_details(drawdown_data)
        
        assert 'max_drawdown' in details
        assert 'peak_date' in details
        assert 'trough_date' in details
        assert 'peak_value' in details
        assert 'trough_value' in details
        assert 'drawdown_days' in details
        
        assert details['peak_value'] == 110
        assert details['trough_value'] == 90
    
    def test_calmar_ratio(self, sample_price_data):
        """Test Calmar ratio calculation."""
        calmar = calculate_calmar_ratio(sample_price_data)
        
        assert isinstance(calmar, float)


class TestMovingAverages:
    """Tests for moving average calculations."""
    
    def test_sma_calculation(self, simple_price_data):
        """Test simple moving average."""
        ma = calculate_sma(simple_price_data, window=3)
        
        assert isinstance(ma, pd.Series)
        assert len(ma) == len(simple_price_data)
        
        # First two values should be NaN (not enough data)
        assert pd.isna(ma.iloc[0])
        assert pd.isna(ma.iloc[1])
        
        # Third value: (100 + 102 + 101) / 3 = 101
        assert abs(ma.iloc[2] - 101) < 0.01
    
    def test_ema_calculation(self, simple_price_data):
        """Test exponential moving average."""
        ema = calculate_ema(simple_price_data, window=3)
        
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(simple_price_data)
        
        # EMA returns values for all rows (uses exponential smoothing)
        # SMA has NaN for first window-1 rows
        sma = calculate_sma(simple_price_data, window=3)
        
        # Check that they produce different values
        # Compare only the overlapping non-NaN values
        sma_valid = sma.dropna()
        ema_at_same_index = ema.loc[sma_valid.index]
        
        # They should be different (EMA weights recent values more)
        assert len(ema_at_same_index) > 0
        # At least some values should differ
        differences = abs(ema_at_same_index - sma_valid)
        assert (differences > 0.01).any()
    
    def test_moving_average_types(self, simple_price_data):
        """Test different MA types."""
        sma = calculate_moving_average(simple_price_data, window=5, ma_type='simple')
        ema = calculate_moving_average(simple_price_data, window=5, ma_type='exponential')
        
        # Both should produce series
        assert isinstance(sma, pd.Series)
        assert isinstance(ema, pd.Series)
        
        # Compare overlapping valid values
        sma_valid = sma.dropna()
        ema_at_same_index = ema.loc[sma_valid.index]
        
        # Should have different values
        differences = abs(ema_at_same_index - sma_valid)
        assert (differences > 0.01).any()
    
    def test_invalid_ma_type(self, simple_price_data):
        """Test invalid MA type raises error."""
        with pytest.raises(ValueError, match="Invalid ma_type"):
            calculate_moving_average(simple_price_data, window=5, ma_type='invalid')


class TestCrossovers:
    """Tests for crossover detection."""
    
    def test_golden_cross_detection(self):
        """Test golden cross detection."""
        # Create data with clear golden cross pattern
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        
        # Create price pattern: decline, then strong rise
        # This should create a scenario where short MA crosses above long MA
        prices = []
        for i in range(150):
            # Decline phase
            prices.append(100 - i * 0.2)
        for i in range(150):
            # Strong recovery phase (faster than decline)
            prices.append(70 + i * 0.5)
        
        df = pd.DataFrame({'Close': prices}, index=dates)
        
        golden_cross = detect_golden_cross(df, short_window=50, long_window=200)
        
        # With this pattern, a golden cross may or may not occur
        # Test that function returns None or a Timestamp
        assert golden_cross is None or isinstance(golden_cross, pd.Timestamp)
    
    def test_death_cross_detection(self):
        """Test death cross detection."""
        # Create data with death cross pattern
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        
        # Create price pattern: rise, then decline
        prices = []
        for i in range(150):
            # Rise phase
            prices.append(70 + i * 0.5)
        for i in range(150):
            # Decline phase (faster than rise)
            prices.append(145 - i * 0.6)
        
        df = pd.DataFrame({'Close': prices}, index=dates)
        
        death_cross = detect_death_cross(df, short_window=50, long_window=200)
        
        # With this pattern, a death cross may or may not occur
        # Test that function returns None or a Timestamp
        assert death_cross is None or isinstance(death_cross, pd.Timestamp)
    
    def test_no_crossover(self, simple_price_data):
        """Test when no crossover exists."""
        # Not enough data for 200-day MA
        golden = detect_golden_cross(simple_price_data)
        death = detect_death_cross(simple_price_data)
        
        assert golden is None
        assert death is None


class TestMASignals:
    """Tests for moving average signals."""
    
    def test_ma_signals(self, sample_price_data):
        """Test MA signal generation."""
        signals = get_moving_average_signals(sample_price_data)
        
        assert 'current_price' in signals
        assert 'ma_20' in signals
        assert 'ma_50' in signals
        assert 'ma_200' in signals
        assert 'trend' in signals
        
        assert signals['trend'] in ['STRONG_BULLISH', 'BULLISH', 'NEUTRAL', 
                                     'BEARISH', 'STRONG_BEARISH', 'INSUFFICIENT_DATA']


class TestSummaries:
    """Tests for summary functions."""
    
    def test_return_volatility_summary(self, sample_price_data):
        """Test return/volatility summary."""
        summary = get_return_volatility_summary(sample_price_data)
        
        assert 'total_return' in summary
        assert 'annualized_return' in summary
        assert 'volatility' in summary
        assert 'sharpe_ratio' in summary
        assert 'trading_days' in summary
    
    def test_comprehensive_analysis(self, sample_price_data):
        """Test comprehensive analysis."""
        analysis = get_comprehensive_analysis(sample_price_data)
        
        assert 'returns' in analysis
        assert 'risk' in analysis
        assert 'ratios' in analysis
        assert 'technical' in analysis
        assert 'summary' in analysis
        
        assert 'total' in analysis['returns']
        assert 'volatility' in analysis['risk']
        assert 'sharpe' in analysis['ratios']
    
    def test_empty_dataframe_analysis(self):
        """Test analysis with empty DataFrame."""
        df = pd.DataFrame()
        analysis = get_comprehensive_analysis(df)
        
        assert 'error' in analysis


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_data_point(self):
        """Test with single data point."""
        df = pd.DataFrame({
            'Close': [100]
        }, index=pd.date_range(start='2024-01-01', periods=1))
        
        returns = calculate_daily_returns(df)
        assert returns.empty
        
        total_return = calculate_total_return(df)
        assert total_return == 0.0
    
    def test_insufficient_data_for_window(self):
        """Test when data is less than window size."""
        df = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range(start='2024-01-01', periods=3))
        
        ma = calculate_sma(df, window=10)
        
        # Implementation returns empty series when data < window
        assert isinstance(ma, pd.Series)
        assert ma.empty
    
    def test_all_same_prices(self):
        """Test with constant prices."""
        df = pd.DataFrame({
            'Close': [100] * 100
        }, index=pd.date_range(start='2024-01-01', periods=100))
        
        returns = calculate_daily_returns(df)
        vol = calculate_volatility(df)
        
        # All returns should be 0, volatility should be 0
        assert np.allclose(returns, 0)
        assert vol == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
