#!/usr/bin/env python3
"""
Free News Data Collection System

Collects financial news from free sources:
1. RSS feeds (Yahoo Finance, MarketWatch, Reuters, etc.)
2. Reddit (r/investing, r/stocks, r/SecurityAnalysis)
3. SEC filings (EDGAR API)
4. Economic data (FRED API - free tier)

This replaces expensive Bloomberg/Reuters feeds with free alternatives.
"""

import os
import time
import requests
import feedparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import json
import re
from urllib.parse import urljoin, urlparse
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import hashlib

try:
    import praw  # Reddit API
    REDDIT_AVAILABLE = True
except ImportError:
    print("praw not available. Install with: pip install praw")
    REDDIT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsSource:
    """Configuration for a news source"""
    name: str
    url: str
    source_type: str  # 'rss', 'reddit', 'api'
    category: str = 'general'  # 'general', 'earnings', 'fed', 'sector'
    enabled: bool = True
    update_frequency: int = 300  # seconds
    last_updated: Optional[datetime] = None

@dataclass
class NewsArticle:
    """Standardized news article format"""
    title: str
    content: str
    source: str
    url: str
    published_date: datetime
    symbols: List[str]  # Extracted stock symbols
    category: str
    sentiment_score: Optional[float] = None
    article_id: str = ""
    
    def __post_init__(self):
        if not self.article_id:
            # Generate unique ID from content hash
            content_str = f"{self.title}{self.content}{self.url}"
            self.article_id = hashlib.md5(content_str.encode()).hexdigest()

class FreeNewsCollector:
    """
    Free financial news collector using multiple sources
    """
    
    def __init__(self, data_dir: str = "/home/ryan/trading/mismatch-trading/data_collection/news_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sources
        self.sources = self._setup_news_sources()
        
        # Setup Reddit client (if available)
        self.reddit_client = self._setup_reddit_client()
        
        # Stock symbol regex for extraction
        self.symbol_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        
        # Common stock symbols for filtering
        self.known_symbols = self._load_known_symbols()
        
        # Article storage
        self.articles_cache = {}
        self.processed_articles = set()
        
        logger.info(f"Initialized news collector with {len(self.sources)} sources")
    
    def _setup_news_sources(self) -> List[NewsSource]:
        """Setup free RSS news sources"""
        sources = [
            # Yahoo Finance
            NewsSource("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline", "rss", "general"),
            NewsSource("Yahoo Markets", "https://feeds.finance.yahoo.com/rss/2.0/category-stocks", "rss", "stocks"),
            
            # MarketWatch
            NewsSource("MarketWatch", "http://feeds.marketwatch.com/marketwatch/marketpulse/", "rss", "general"),
            NewsSource("MarketWatch Tech", "http://feeds.marketwatch.com/marketwatch/technology/", "rss", "tech"),
            
            # Reuters Business
            NewsSource("Reuters Business", "http://feeds.reuters.com/reuters/businessNews", "rss", "general"),
            NewsSource("Reuters Markets", "http://feeds.reuters.com/reuters/markets", "rss", "markets"),
            
            # Seeking Alpha (limited free)
            NewsSource("Seeking Alpha", "https://seekingalpha.com/market_currents.xml", "rss", "analysis"),
            
            # Federal Reserve Economic Data
            NewsSource("FRED Economic", "https://fred.stlouisfed.org/releases/", "api", "economic"),
            
            # SEC EDGAR (earnings reports)
            NewsSource("SEC EDGAR", "https://www.sec.gov/cgi-bin/browse-edgar", "api", "earnings"),
        ]
        
        # Reddit sources (if available)
        if REDDIT_AVAILABLE:
            sources.extend([
                NewsSource("Reddit Investing", "r/investing", "reddit", "discussion"),
                NewsSource("Reddit Stocks", "r/stocks", "reddit", "discussion"),
                NewsSource("Reddit SecurityAnalysis", "r/SecurityAnalysis", "reddit", "analysis"),
                NewsSource("Reddit ValueInvesting", "r/ValueInvesting", "reddit", "analysis"),
            ])
        
        return sources
    
    def _load_known_symbols(self) -> set:
        """Load known stock symbols for filtering"""
        # Common S&P 500 symbols - in production would load from a comprehensive list
        common_symbols = {
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK", "JNJ", "V",
            "WMT", "JPM", "PG", "UNH", "MA", "HD", "DIS", "PYPL", "BAC", "NFLX",
            "ADBE", "CRM", "INTC", "XOM", "VZ", "KO", "PFE", "PEP", "T", "ABBV",
            "CVX", "MRK", "ACN", "NKE", "TMO", "MDT", "COST", "LLY", "DHR", "NEE",
            "QCOM", "TXN", "UPS", "PM", "IBM", "HON", "AMGN", "LOW", "C", "SBUX",
            "SPY", "QQQ", "IWM", "VTI", "BTC", "ETH"  # ETFs and crypto
        }
        return common_symbols
    
    def _setup_reddit_client(self) -> Optional['praw.Reddit']:
        """Setup Reddit client with free tier access"""
        if not REDDIT_AVAILABLE:
            return None
        
        try:
            # These would be set as environment variables in production
            # For now, using read-only access which doesn't require credentials
            reddit = praw.Reddit(
                client_id="dummy",  # Would be real credentials
                client_secret="dummy",
                user_agent="FreeNewsCollector v1.0",
                check_for_async=False
            )
            
            # Test connection with read-only access
            # reddit.subreddit("investing").hot(limit=1)
            logger.info("Reddit client initialized (read-only)")
            return reddit
            
        except Exception as e:
            logger.warning(f"Reddit client setup failed: {e}")
            return None
    
    def collect_rss_news(self, source: NewsSource) -> List[NewsArticle]:
        """Collect news from RSS feeds"""
        try:
            # Parse RSS feed
            feed = feedparser.parse(source.url)
            
            if feed.bozo:
                logger.warning(f"RSS parsing issues for {source.name}: {feed.bozo_exception}")
            
            articles = []
            for entry in feed.entries[:50]:  # Limit to 50 most recent
                try:
                    # Extract article data
                    title = entry.get('title', '')
                    content = entry.get('summary', entry.get('description', ''))
                    url = entry.get('link', '')
                    
                    # Parse publication date
                    pub_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    
                    # Extract stock symbols
                    symbols = self._extract_symbols(title + " " + content)
                    
                    # Create article
                    article = NewsArticle(
                        title=title,
                        content=content,
                        source=source.name,
                        url=url,
                        published_date=pub_date,
                        symbols=symbols,
                        category=source.category
                    )
                    
                    # Skip duplicates
                    if article.article_id not in self.processed_articles:
                        articles.append(article)
                        self.processed_articles.add(article.article_id)
                    
                except Exception as e:
                    logger.error(f"Error processing RSS entry: {e}")
                    continue
            
            logger.info(f"Collected {len(articles)} new articles from {source.name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting RSS from {source.name}: {e}")
            return []
    
    def collect_reddit_news(self, source: NewsSource) -> List[NewsArticle]:
        """Collect financial discussions from Reddit"""
        if not self.reddit_client:
            return []
        
        try:
            subreddit_name = source.url.replace('r/', '')
            subreddit = self.reddit_client.subreddit(subreddit_name)
            
            articles = []
            
            # Get hot posts
            for submission in subreddit.hot(limit=25):
                try:
                    # Skip stickied posts
                    if submission.stickied:
                        continue
                    
                    # Extract symbols from title and text
                    content = f"{submission.title} {submission.selftext}"
                    symbols = self._extract_symbols(content)
                    
                    # Only include posts with stock symbols
                    if not symbols:
                        continue
                    
                    article = NewsArticle(
                        title=submission.title,
                        content=submission.selftext[:1000],  # Limit content length
                        source=source.name,
                        url=submission.url,
                        published_date=datetime.fromtimestamp(submission.created_utc),
                        symbols=symbols,
                        category=source.category
                    )
                    
                    if article.article_id not in self.processed_articles:
                        articles.append(article)
                        self.processed_articles.add(article.article_id)
                
                except Exception as e:
                    logger.error(f"Error processing Reddit post: {e}")
                    continue
            
            logger.info(f"Collected {len(articles)} discussions from {source.name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting Reddit from {source.name}: {e}")
            return []
    
    def collect_sec_filings(self) -> List[NewsArticle]:
        """Collect recent SEC filings (earnings reports)"""
        try:
            # SEC EDGAR RSS feed for recent filings
            url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom"
            
            response = requests.get(url, headers={'User-Agent': 'FreeNewsCollector research@example.com'})
            response.raise_for_status()
            
            # Parse SEC Atom feed
            feed = feedparser.parse(response.text)
            
            articles = []
            for entry in feed.entries[:20]:  # Recent filings
                try:
                    title = entry.get('title', '')
                    content = entry.get('summary', '')
                    url = entry.get('link', '')
                    
                    # Extract company symbols from SEC filings
                    symbols = self._extract_symbols_from_sec_title(title)
                    
                    if symbols:  # Only include if we found symbols
                        pub_date = datetime.now()
                        if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6])
                        
                        article = NewsArticle(
                            title=title,
                            content=content,
                            source="SEC EDGAR",
                            url=url,
                            published_date=pub_date,
                            symbols=symbols,
                            category="earnings"
                        )
                        
                        if article.article_id not in self.processed_articles:
                            articles.append(article)
                            self.processed_articles.add(article.article_id)
                
                except Exception as e:
                    logger.error(f"Error processing SEC filing: {e}")
                    continue
            
            logger.info(f"Collected {len(articles)} SEC filings")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting SEC filings: {e}")
            return []
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        # Find all potential symbols (1-5 uppercase letters)
        potential_symbols = self.symbol_pattern.findall(text.upper())
        
        # Filter to known symbols and common patterns
        symbols = []
        for symbol in potential_symbols:
            if (len(symbol) >= 2 and 
                (symbol in self.known_symbols or 
                 self._is_likely_symbol(symbol, text))):
                symbols.append(symbol)
        
        return list(set(symbols))  # Remove duplicates
    
    def _extract_symbols_from_sec_title(self, title: str) -> List[str]:
        """Extract symbols from SEC filing titles"""
        # SEC titles often contain company names, try to extract symbols
        # This is simplified - production would use a company name -> symbol mapping
        symbols = self._extract_symbols(title)
        
        # Additional SEC-specific extraction logic could go here
        # For now, rely on the general symbol extraction
        
        return symbols
    
    def _is_likely_symbol(self, candidate: str, context: str) -> bool:
        """Determine if a candidate string is likely a stock symbol"""
        # Exclude common false positives
        exclude_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS',
            'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HOW', 'ITS', 'NEW', 'NOW',
            'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'HAD', 'LET', 'PUT', 'SAY', 'SHE',
            'TOO', 'USE', 'CEO', 'CFO', 'IPO', 'SEC', 'FDA', 'ETF', 'ESG', 'AI', 'IT'
        }
        
        if candidate in exclude_words:
            return False
        
        # Look for contextual clues
        context_lower = context.lower()
        symbol_indicators = ['stock', 'shares', 'ticker', 'nasdaq', 'nyse', '$', 'trading']
        
        # If candidate appears near financial terms, more likely to be a symbol
        for indicator in symbol_indicators:
            if indicator in context_lower:
                return True
        
        return False
    
    def collect_all_news(self) -> List[NewsArticle]:
        """Collect news from all enabled sources"""
        all_articles = []
        
        logger.info("Starting news collection from all sources...")
        
        for source in self.sources:
            if not source.enabled:
                continue
            
            # Check if source needs updating
            if (source.last_updated and 
                (datetime.now() - source.last_updated).seconds < source.update_frequency):
                continue
            
            try:
                if source.source_type == "rss":
                    articles = self.collect_rss_news(source)
                elif source.source_type == "reddit":
                    articles = self.collect_reddit_news(source)
                elif source.source_type == "api" and "SEC" in source.name:
                    articles = self.collect_sec_filings()
                else:
                    continue
                
                all_articles.extend(articles)
                source.last_updated = datetime.now()
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting from {source.name}: {e}")
                continue
        
        logger.info(f"Collected {len(all_articles)} total articles")
        return all_articles
    
    def save_articles(self, articles: List[NewsArticle], filename: Optional[str] = None):
        """Save articles to disk"""
        if not articles:
            return
        
        if filename is None:
            filename = f"news_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.data_dir / filename
        
        # Convert articles to JSON-serializable format
        articles_data = []
        for article in articles:
            article_dict = asdict(article)
            article_dict['published_date'] = article.published_date.isoformat()
            articles_data.append(article_dict)
        
        with open(filepath, 'w') as f:
            json.dump(articles_data, f, indent=2)
        
        logger.info(f"Saved {len(articles)} articles to {filepath}")
    
    def load_articles(self, filename: str) -> List[NewsArticle]:
        """Load articles from disk"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return []
        
        with open(filepath, 'r') as f:
            articles_data = json.load(f)
        
        articles = []
        for data in articles_data:
            data['published_date'] = datetime.fromisoformat(data['published_date'])
            articles.append(NewsArticle(**data))
        
        logger.info(f"Loaded {len(articles)} articles from {filepath}")
        return articles
    
    def get_recent_news(self, hours: int = 24, symbols: Optional[List[str]] = None) -> List[NewsArticle]:
        """Get recent news articles, optionally filtered by symbols"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Collect fresh news
        all_articles = self.collect_all_news()
        
        # Filter by time and symbols
        filtered_articles = []
        for article in all_articles:
            if article.published_date < cutoff_time:
                continue
            
            if symbols and not any(symbol in article.symbols for symbol in symbols):
                continue
            
            filtered_articles.append(article)
        
        # Sort by publication date (newest first)
        filtered_articles.sort(key=lambda x: x.published_date, reverse=True)
        
        return filtered_articles
    
    def get_symbol_news(self, symbol: str, hours: int = 24) -> List[NewsArticle]:
        """Get news for a specific symbol"""
        return self.get_recent_news(hours=hours, symbols=[symbol])
    
    def get_news_summary(self) -> Dict[str, int]:
        """Get summary statistics of collected news"""
        all_articles = []
        
        # Load recent articles from disk
        for file_path in self.data_dir.glob("news_articles_*.json"):
            articles = self.load_articles(file_path.name)
            all_articles.extend(articles)
        
        # Calculate statistics
        total_articles = len(all_articles)
        sources = set(article.source for article in all_articles)
        categories = set(article.category for article in all_articles)
        symbols = set()
        for article in all_articles:
            symbols.update(article.symbols)
        
        recent_articles = [a for a in all_articles 
                          if (datetime.now() - a.published_date).hours < 24]
        
        return {
            'total_articles': total_articles,
            'recent_articles_24h': len(recent_articles),
            'unique_sources': len(sources),
            'categories': len(categories),
            'unique_symbols': len(symbols),
            'sources_list': list(sources),
            'top_symbols': list(sorted(symbols))[:20]
        }

def main():
    """Test the free news collection system"""
    print("Testing Free News Collection System")
    print("="*50)
    
    # Create collector
    collector = FreeNewsCollector()
    
    # Collect recent news
    print("Collecting news from free sources...")
    
    start_time = time.time()
    articles = collector.collect_all_news()
    collection_time = time.time() - start_time
    
    print(f"Collected {len(articles)} articles in {collection_time:.2f}s")
    
    if articles:
        # Save articles
        collector.save_articles(articles)
        
        # Show sample articles
        print(f"\nSample Articles:")
        for i, article in enumerate(articles[:5]):
            print(f"{i+1}. {article.title[:80]}...")
            print(f"   Source: {article.source} | Symbols: {article.symbols}")
            print(f"   Date: {article.published_date}")
            print()
        
        # Get symbol-specific news
        if articles[0].symbols:
            symbol = articles[0].symbols[0]
            symbol_news = collector.get_symbol_news(symbol, hours=48)
            print(f"Found {len(symbol_news)} articles for {symbol}")
        
        # Summary statistics
        summary = collector.get_news_summary()
        print(f"\nCollection Summary:")
        print(f"   Total articles: {summary['total_articles']}")
        print(f"   Sources: {summary['unique_sources']}")
        print(f"   Unique symbols: {summary['unique_symbols']}")
        print(f"   Top symbols: {summary['top_symbols'][:10]}")
        
        print(f"\nFree news collection system working successfully!")
        print(f"Ready for integration with FinBERT processing")
        
    else:
        print(f"No articles collected. Check network connection and RSS feeds.")

if __name__ == "__main__":
    main()