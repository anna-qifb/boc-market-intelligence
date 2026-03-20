import streamlit as st
import requests
from datetime import datetime, timedelta
import pytz
from transformers import pipeline
import matplotlib.pyplot as plt
import re
from collections import Counter

st.set_page_config(page_title="HSBC Market Intelligence", layout="wide")
st.title("HSBC – Deep Learning Market Intelligence")
st.markdown("### Real-time Financial News Sentiment Analysis & Recommendation")

# Load models (cached)
@st.cache_resource
def load_models():
    sentiment = pipeline("text-classification", model="fongkw2025/finbert-sentiment-jpm")
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
    return sentiment, generator

with st.spinner("Loading AI models..."):
    sentiment_pipe, gen_pipe = load_models()

def get_time_from():
    now_utc = datetime.now(pytz.utc)
    return (now_utc - timedelta(hours=72)).strftime('%Y%m%dT%H%M')

def extract_key_points(articles):
    text = " ".join([a.get('title', '') + " " + a.get('summary', '') for a in articles])
    if not text.strip():
        return "No key themes detected"
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'will', 'have', 'says', 'said', 'been', 'are',
            'was', 'which', 'their', 'over', 'more', 'year', 'also', 'after', 'into', 'has', 'its'}
    common = Counter(w for w in words if w not in stop).most_common(10)
    key_words = [w for w, c in common if c > 1]
    return ", ".join(key_words[:8]) or "general market updates"

# User input
ticker = st.text_input("Enter U.S. Stock Ticker (e.g., AAPL, NVDA, TSLA):", placeholder="e.g., AAPL").upper().strip()

if st.button("Generate Market Intelligence Report", type="primary"):
    if not ticker:
        st.error("Please enter a valid ticker symbol.")
    else:
        with st.spinner("Fetching real-time news and analyzing..."):
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={get_time_from()}&limit=10&topics=financial_markets,finance&sort=LATEST&apikey=WW21DG3S70HJR18U"
            data = requests.get(url).json()

            if 'feed' not in data or not data['feed']:
                st.warning(f"No recent news found for {ticker}.")
                st.info("**Recommendation: Hold** (insufficient data)")
                st.stop()

            # Extract and filter high-relevance articles (>= 0.7)
            articles = []
            for a in data['feed']:
                relevance = 0.0
                for ts in a.get('ticker_sentiment', []):
                    if ts.get('ticker') == ticker:
                        try:
                            relevance = float(ts.get('relevance_score', 0))
                        except:
                            relevance = 0.0
                        break
                if relevance >= 0.7:
                    articles.append({**a, 'relevance_score': relevance})

            if not articles:
                st.warning(f"No highly relevant news for {ticker}.")
                st.info("**Recommendation: Hold**")
                st.stop()

            # Sort by relevance and take top 5
            articles.sort(key=lambda x: x['relevance_score'], reverse=True)
            top_articles = articles[:5]

            # Sentiment analysis + store per-article results
            weighted_scores = []
            total_conf = 0.0
            sentiment_labels = []
            article_details = []  # For display in expander

            for a in top_articles:
                summary = a.get('summary', '').strip()
                if summary:
                    result = sentiment_pipe(summary)[0]
                    label = result['label'].lower()  # normalize
                    conf = result['score']
                    score_map = {'positive': 2, 'neutral': 1, 'negative': 0}
                    score = score_map.get(label, 1)

                    weighted_scores.append(score * conf)
                    total_conf += conf
                    sentiment_labels.append(label)
                    article_details.append({
                        'sentiment': label.capitalize(),
                        'confidence': conf
                    })
                else:
                    # Handle articles without summary
                    weighted_scores.append(1.0 * 0.5)
                    total_conf += 0.5
                    sentiment_labels.append('neutral')
                    article_details.append({
                        'sentiment': 'Neutral',
                        'confidence': 0.5
                    })

            aggregated_score = sum(weighted_scores) / total_conf if total_conf > 0 else 1.0
            avg_relevance = sum(a['relevance_score'] for a in top_articles) / len(top_articles)

            # Recommendation logic
            base_rec = "Buy" if aggregated_score >= 1.5 else "Hold" if aggregated_score >= 0.5 else "Sell"
            recommendation = (
                f"Strong {base_rec}" if avg_relevance >= 0.85 else
                f"Cautious {base_rec}" if avg_relevance >= 0.7 else
                base_rec
            )

            # Sentiment distribution
            dist = Counter(sentiment_labels)
            total = len(sentiment_labels)
            pos = (dist.get('positive', 0) / total) * 100 if total > 0 else 0
            neu = (dist.get('neutral', 0) / total) * 100 if total > 0 else 0
            neg = (dist.get('negative', 0) / total) * 100 if total > 0 else 0

            # Display layout
            col1, col2 = st.columns([1, 2])
            with col1:
                fig, ax = plt.subplots(figsize=(4, 4))
                sizes = [pos, neu, neg]
                if sum(sizes) == 0:
                    sizes = [0, 100, 0]
                ax.pie(sizes,
                       labels=['Positive', 'Neutral', 'Negative'],
                       autopct='%1.1f%%',
                       colors=['#2ecc71', '#f1c40f', '#e74c3c'],
                       startangle=90,
                       textprops={'fontsize': 10})
                ax.set_title("Sentiment Distribution", fontsize=12, pad=20)
                st.pyplot(fig)

            with col2:
                st.metric("Aggregated Sentiment Score", f"{aggregated_score:.2f}")
                st.metric("Average Relevance Score", f"{avg_relevance:.2f}")
                st.success(f"**Final Recommendation: {recommendation}**")

            key_points = extract_key_points(top_articles)

            # Professional report
            structured_report = f"""
**Ticker:** {ticker}  
*As of {datetime.now().strftime('%B %d, %Y')}*
**Market Sentiment Overview**  
Recent financial news coverage over the past 72 hours indicates the following sentiment distribution among highly relevant articles:  
• **Positive**: {pos:.1f}%  
• **Neutral**: {neu:.1f}%  
• **Negative**: {neg:.1f}%  
The aggregated sentiment score stands at **{aggregated_score:.2f}** (on a scale where higher values reflect more positive outlook), with an average news relevance of **{avg_relevance:.2f}** (indicating strong company-specific coverage).
**Key Thematic Insights**  
Prominent themes emerging from recent coverage include: **{key_points}**.
**Investment Recommendation**  
Based on the quantitative sentiment analysis and relevance-weighted assessment, our recommendation is to **{recommendation}**.
This signal reflects a balanced view of current market narrative. Investors should consider this alongside fundamental analysis, portfolio objectives, and risk tolerance. Market conditions can change rapidly.
*This report is generated using AI-driven sentiment analysis and is for informational purposes only.*
            """
            
            st.write(structured_report)
            
            prompt = f"Write a professional investment recommendation report for {ticker} stock. Use a formal financial advisor tone."
            
            with st.spinner("Generating enhanced report with AI text generation..."):
                try:
                    generated = gen_pipe(prompt, max_length=500, temperature=0.8, do_sample=True, truncation=True)[0]['generated_text']
                    st.markdown("### Investment Recommendation Report")
                    st.write(generated)
                except Exception as e:
                    st.warning(f"Could not generate AI-enhanced report: {str(e)}")

            # Source articles with per-article sentiment
            with st.expander("View Source News Articles"):
                for i, a in enumerate(top_articles, 1):
                    details = article_details[i-1]
                    st.subheader(f"Article {i}: {a.get('title', 'No title')}")
                    st.write(a.get('summary', 'No summary'))
                    st.caption(f"Source: [{a.get('source', 'Unknown')}]({a.get('url', '#')})")
                    st.caption(f"Relevance Score: {a['relevance_score']:.2f} | "
                               f"Sentiment: {details['sentiment']} (confidence: {details['confidence']:.2f})")
                    st.divider()
