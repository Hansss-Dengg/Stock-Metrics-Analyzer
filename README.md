# Stock Performance Analyzer

Python app to fetch historical stock prices, compute returns, volatility, drawdown, and visualize results.

## Tech Stack
- Python, pandas, yfinance, plotly
- Streamlit (optional)
- Yahoo Finance API

## Installation

```bash
# Clone repository
git clone https://github.com/Hansss-Dengg/stock-performance-analyzer.git
cd stock-performance-analyzer

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Application (Streamlit)

Launch the interactive web dashboard:

```bash
python run_app.py
```

The app will open in your browser at http://localhost:8501

### Python API

```python
from spa.data_fetcher import fetch_stock_data
from spa.data_processor import calculate_comprehensive_analysis
from spa.visualizer import create_price_chart

# Fetch stock data
df = fetch_stock_data('AAPL', period='1y')

# Analyze performance
analysis = calculate_comprehensive_analysis(df)
print(f"Total Return: {analysis['returns']['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {analysis['ratios']['sharpe_ratio']:.2f}")

# Create interactive chart
fig = create_price_chart(df, ticker='AAPL')
fig.show()
```

## Running Tests

```bash
pytest
```

## Features

### Data Management
- Fetch historical stock data from Yahoo Finance
- Automatic caching with 24-hour TTL (90% API call reduction)
- Retry logic with exponential backoff for network errors
- Comprehensive error handling and validation

### Financial Metrics (30+ calculations)
- Returns: daily, cumulative, annualized
- Volatility: standard, rolling, downside
- Drawdown: current, maximum, recovery analysis
- Risk ratios: Sharpe, Sortino, Calmar
- Moving averages: SMA, EMA with golden/death cross detection

### Visualizations (6+ interactive charts)
- Candlestick price charts with volume
- Daily and cumulative returns
- Rolling volatility analysis
- Drawdown tracking with max DD highlighting
- Moving average overlays with crossover markers
- Multi-stock comparison (normalized price, returns, volatility)

### Web Dashboard
- Interactive Streamlit interface
- Real-time stock data fetching
- Multiple analysis pages (Overview, Price, Returns, Volatility, Drawdown, Technical, Comparison)
- Customizable date ranges and parameters
- Multi-stock comparison tool
- Export functionality (CSV, JSON, HTML charts, text reports)
- Performance optimizations with caching
- Progress indicators and error handling

## Deployment

### Streamlit Community Cloud

The app is deployed on Streamlit Community Cloud. To deploy your own instance:

1. **Fork the repository** on GitHub
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Sign in** with your GitHub account
4. **Click** "New app"
5. **Select** your forked repository
6. **Set** the main file path: `src/spa/app.py`
7. **Click** "Deploy"

The app will be live at: `https://[your-username]-stock-performance-analyzer.streamlit.app`

### Configuration Files

- `.streamlit/config.toml` - Streamlit configuration
- `requirements.txt` - Python dependencies
- `packages.txt` - System dependencies (if needed)
- `.python-version` - Python version specification

## Project Structure

```
stock-performance-analyzer/
├── src/spa/              # Main application package
│   ├── app.py           # Streamlit web application
│   ├── data_fetcher.py  # Yahoo Finance API integration
│   ├── data_processor.py # Financial metrics calculations
│   ├── data_cleaner.py  # Data validation and cleaning
│   ├── visualizer.py    # Plotly chart creation
│   ├── cache.py         # File-based caching
│   ├── retry.py         # Retry logic with backoff
│   └── exceptions.py    # Custom exceptions
├── tests/               # Unit tests
├── .streamlit/          # Streamlit configuration
├── requirements.txt     # Python dependencies
├── run_app.py          # Local development launcher
└── README.md           # Documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
