"""
Data cleaning module for stock market data.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def prepare_dataframe(
    data: Union[pd.DataFrame, Dict, List],
    date_column: Optional[str] = None,
    ensure_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert various data formats into a standardized pandas DataFrame.
    
    Args:
        data: Input data (DataFrame, dict, or list of dicts)
        date_column: Name of column containing dates (if not index)
        ensure_columns: List of required columns to validate
    
    Returns:
        Standardized DataFrame with DatetimeIndex
    
    Raises:
        ValueError: If data cannot be converted or lacks required columns
    """
    # Convert to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame.from_dict(data, orient='index')
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    if df.empty:
        raise ValueError("Cannot prepare empty DataFrame")
    
    # Set datetime index if specified
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"Cannot convert index to datetime: {e}")
    
    # Validate required columns
    if ensure_columns:
        missing = [col for col in ensure_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame to standard format for stock data.
    
    Ensures:
    - DatetimeIndex sorted ascending
    - Standard column names
    - Numeric data types
    - No duplicate indices
    
    Args:
        df: Input DataFrame
    
    Returns:
        Normalized DataFrame
    """
    if df.empty:
        return df
    
    df_norm = df.copy()
    
    # Ensure DatetimeIndex
    if not isinstance(df_norm.index, pd.DatetimeIndex):
        df_norm.index = pd.to_datetime(df_norm.index)
    
    # Sort by date
    df_norm = df_norm.sort_index()
    
    # Standardize column names
    df_norm = standardize_columns(df_norm)
    
    # Remove duplicate indices
    df_norm = df_norm[~df_norm.index.duplicated(keep='first')]
    
    # Convert numeric columns to float
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']
    for col in numeric_cols:
        if col in df_norm.columns:
            df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce')
    
    # Set index name if not already set
    if df_norm.index.name is None:
        df_norm.index.name = 'Date'
    
    return df_norm


def extract_price_data(
    df: pd.DataFrame,
    price_column: str = 'Close'
) -> pd.Series:
    """
    Extract a single price column as a Series.
    
    Args:
        df: Input DataFrame
        price_column: Column name to extract (default: 'Close')
    
    Returns:
        Series with price data
    
    Raises:
        ValueError: If column doesn't exist
    """
    if price_column not in df.columns:
        # Try alternative names
        alt_names = {
            'Close': ['close', 'Close Price', 'close_price'],
            'Adj_Close': ['Adj Close', 'adj close', 'adjusted_close']
        }
        
        if price_column in alt_names:
            for alt in alt_names[price_column]:
                if alt in df.columns:
                    price_column = alt
                    break
            else:
                raise ValueError(f"Column '{price_column}' not found in DataFrame")
        else:
            raise ValueError(f"Column '{price_column}' not found in DataFrame")
    
    return df[price_column].copy()


def resample_data(
    df: pd.DataFrame,
    frequency: str = 'D',
    aggregation: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Resample data to different frequency (daily, weekly, monthly).
    
    Args:
        df: Input DataFrame with DatetimeIndex
        frequency: Resampling frequency ('D', 'W', 'M', etc.)
        aggregation: Dictionary mapping columns to aggregation methods.
                    If None, uses defaults (OHLC for prices, sum for volume)
    
    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df
    
    if aggregation is None:
        # Default aggregations
        aggregation = {}
        if 'Open' in df.columns:
            aggregation['Open'] = 'first'
        if 'High' in df.columns:
            aggregation['High'] = 'max'
        if 'Low' in df.columns:
            aggregation['Low'] = 'min'
        if 'Close' in df.columns:
            aggregation['Close'] = 'last'
        if 'Adj_Close' in df.columns:
            aggregation['Adj_Close'] = 'last'
        if 'Volume' in df.columns:
            aggregation['Volume'] = 'sum'
    
    df_resampled = df.resample(frequency).agg(aggregation)
    
    # Remove rows with all NaN
    df_resampled = df_resampled.dropna(how='all')
    
    return df_resampled


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame, keeping the first occurrence.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    initial_rows = len(df)
    df_clean = df[~df.index.duplicated(keep='first')]
    
    removed = initial_rows - len(df_clean)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")
    
    return df_clean


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required columns and proper structure.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if data is valid
    
    Raises:
        ValueError: If data is invalid
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for required columns (flexible - at least Close should exist)
    required_cols = ['Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Check for numeric data
    if not df['Close'].dtype in [np.float64, np.float32, np.int64, np.int32]:
        raise ValueError("Close column must contain numeric data")
    
    return True


def remove_invalid_prices(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove rows with invalid price data (negative, zero, or NaN).
    
    Args:
        df: Input DataFrame
        columns: List of columns to check. If None, checks common price columns.
    
    Returns:
        DataFrame with invalid prices removed
    """
    if df.empty:
        return df
    
    if columns is None:
        # Check common price columns that exist in the DataFrame
        columns = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df.columns]
    
    initial_rows = len(df)
    df_clean = df.copy()
    
    # Remove rows where any price column is <= 0 or NaN
    for col in columns:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col] > 0]
            df_clean = df_clean[df_clean[col].notna()]
    
    removed = initial_rows - len(df_clean)
    if removed > 0:
        logger.warning(f"Removed {removed} rows with invalid prices")
    
    return df_clean


def remove_outliers(
    df: pd.DataFrame, 
    column: str = 'Close',
    std_threshold: float = 4.0
) -> pd.DataFrame:
    """
    Remove statistical outliers using standard deviation method.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        std_threshold: Number of standard deviations for outlier detection (default: 4)
    
    Returns:
        DataFrame with outliers removed
    """
    if df.empty or column not in df.columns:
        return df
    
    initial_rows = len(df)
    
    # Calculate mean and std
    mean = df[column].mean()
    std = df[column].std()
    
    # Define outlier bounds
    lower_bound = mean - (std_threshold * std)
    upper_bound = mean + (std_threshold * std)
    
    # Filter outliers
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    removed = initial_rows - len(df_clean)
    if removed > 0:
        logger.info(f"Removed {removed} outliers from {column}")
    
    return df_clean


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and ensure consistent format.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with standardized column names
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Common column name mappings
    column_mapping = {
        'Adj Close': 'Adj_Close',
        'adj close': 'Adj_Close',
        'adjusted close': 'Adj_Close',
    }
    
    # Apply mappings if columns exist
    for old_name, new_name in column_mapping.items():
        if old_name in df_clean.columns:
            df_clean.rename(columns={old_name: new_name}, inplace=True)
    
    return df_clean


def sort_by_date(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """
    Sort DataFrame by date index.
    
    Args:
        df: Input DataFrame
        ascending: Sort in ascending order (True) or descending (False)
    
    Returns:
        Sorted DataFrame
    """
    if df.empty:
        return df
    
    return df.sort_index(ascending=ascending)


def fill_missing_data(
    df: pd.DataFrame,
    method: str = 'ffill',
    limit: Optional[int] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fill missing data using various interpolation methods.
    
    Args:
        df: Input DataFrame
        method: Fill method - 'ffill' (forward fill), 'bfill' (backward fill),
                'interpolate' (linear), or 'drop'
        limit: Maximum number of consecutive NaNs to fill
        columns: Specific columns to fill. If None, fills all numeric columns.
    
    Returns:
        DataFrame with missing data filled
    """
    if df.empty:
        return df
    
    df_filled = df.copy()
    
    # Determine which columns to fill
    if columns is None:
        columns = df_filled.select_dtypes(include=[np.number]).columns.tolist()
    
    initial_nans = df_filled[columns].isna().sum().sum()
    
    if method == 'ffill':
        # Forward fill - use last known value
        df_filled[columns] = df_filled[columns].fillna(method='ffill', limit=limit)
    
    elif method == 'bfill':
        # Backward fill - use next known value
        df_filled[columns] = df_filled[columns].fillna(method='bfill', limit=limit)
    
    elif method == 'interpolate':
        # Linear interpolation
        df_filled[columns] = df_filled[columns].interpolate(method='linear', limit=limit)
    
    elif method == 'drop':
        # Drop rows with any NaN
        df_filled = df_filled.dropna(subset=columns)
    
    else:
        raise ValueError(f"Unknown fill method: {method}")
    
    filled_nans = initial_nans - df_filled[columns].isna().sum().sum()
    if filled_nans > 0:
        logger.info(f"Filled {filled_nans} missing values using {method}")
    
    return df_filled


def handle_data_gaps(
    df: pd.DataFrame,
    max_gap_days: int = 5,
    fill_method: str = 'interpolate'
) -> pd.DataFrame:
    """
    Handle gaps in time series data (e.g., weekends, holidays).
    
    Args:
        df: Input DataFrame with DatetimeIndex
        max_gap_days: Maximum gap size to fill (larger gaps are left as-is)
        fill_method: Method to use for filling gaps
    
    Returns:
        DataFrame with gaps handled
    """
    if df.empty or len(df) < 2:
        return df
    
    df_filled = df.copy()
    
    # Identify gaps
    date_diffs = df_filled.index.to_series().diff()
    gap_mask = date_diffs > pd.Timedelta(days=max_gap_days)
    
    num_gaps = gap_mask.sum()
    if num_gaps > 0:
        logger.info(f"Found {num_gaps} gaps larger than {max_gap_days} days")
    
    # Fill smaller gaps
    df_filled = fill_missing_data(df_filled, method=fill_method)
    
    return df_filled


def detect_missing_dates(
    df: pd.DataFrame,
    business_days_only: bool = True
) -> List[datetime]:
    """
    Detect missing dates in the time series.
    
    Args:
        df: Input DataFrame with DatetimeIndex
        business_days_only: If True, only considers business days (Mon-Fri)
    
    Returns:
        List of missing dates
    """
    if df.empty or len(df) < 2:
        return []
    
    start_date = df.index.min()
    end_date = df.index.max()
    
    if business_days_only:
        # Generate business day range
        expected_dates = pd.bdate_range(start=start_date, end=end_date)
    else:
        # Generate daily range
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Find missing dates
    missing = expected_dates.difference(df.index)
    
    if len(missing) > 0:
        logger.info(f"Found {len(missing)} missing dates")
    
    return missing.tolist()


def fill_missing_dates(
    df: pd.DataFrame,
    business_days_only: bool = True,
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    """
    Fill in missing dates with appropriate values.
    
    Args:
        df: Input DataFrame with DatetimeIndex
        business_days_only: If True, only fills business days
        fill_method: Method to use for filling values
    
    Returns:
        DataFrame with complete date range
    """
    if df.empty:
        return df
    
    start_date = df.index.min()
    end_date = df.index.max()
    
    if business_days_only:
        complete_dates = pd.bdate_range(start=start_date, end=end_date)
    else:
        complete_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex to complete date range
    df_complete = df.reindex(complete_dates)
    
    # Fill missing values
    df_complete = fill_missing_data(df_complete, method=fill_method)
    
    return df_complete


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a report on data quality metrics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with quality metrics
    """
    if df.empty:
        return {'status': 'empty'}
    
    report = {
        'total_rows': len(df),
        'date_range': {
            'start': df.index.min(),
            'end': df.index.max(),
            'days': (df.index.max() - df.index.min()).days
        },
        'missing_values': {},
        'duplicate_dates': df.index.duplicated().sum(),
        'columns': list(df.columns)
    }
    
    # Count missing values per column
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            report['missing_values'][col] = {
                'count': int(nan_count),
                'percentage': round(nan_count / len(df) * 100, 2)
            }
    
    # Detect gaps
    missing_dates = detect_missing_dates(df, business_days_only=True)
    report['missing_dates'] = len(missing_dates)
    
    return report


def clean_stock_data(
    df: pd.DataFrame,
    remove_dups: bool = True,
    remove_invalid: bool = True,
    remove_outliers_flag: bool = False,
    std_threshold: float = 4.0
) -> pd.DataFrame:
    """
    Comprehensive cleaning pipeline for stock data.
    
    Args:
        df: Input DataFrame
        remove_dups: Remove duplicate rows
        remove_invalid: Remove invalid price data
        remove_outliers_flag: Remove statistical outliers
        std_threshold: Standard deviation threshold for outlier removal
    
    Returns:
        Cleaned DataFrame
    
    Raises:
        ValueError: If data validation fails
    """
    if df.empty:
        logger.warning("Received empty DataFrame")
        return df
    
    logger.info(f"Cleaning data: {len(df)} rows")
    
    df_clean = df.copy()
    
    # Standardize column names
    df_clean = standardize_columns(df_clean)
    
    # Sort by date
    df_clean = sort_by_date(df_clean)
    
    # Remove duplicates
    if remove_dups:
        df_clean = remove_duplicates(df_clean)
    
    # Remove invalid prices
    if remove_invalid:
        df_clean = remove_invalid_prices(df_clean)
    
    # Remove outliers (optional)
    if remove_outliers_flag:
        df_clean = remove_outliers(df_clean, std_threshold=std_threshold)
    
    # Validate final result
    validate_data(df_clean)
    
    logger.info(f"Cleaning complete: {len(df_clean)} rows remaining")
    
    return df_clean
