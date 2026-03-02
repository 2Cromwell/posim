"""
Visualization module for annotation results
All labels and text in English
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Configure Chinese font support
import platform
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class AnnotationVisualizer:
    def __init__(self, labels_path: str = "data_process/xibeiyuzhicai/output/labels.json"):
        self.labels_path = Path(labels_path)
        self.output_dir = self.labels_path.parent / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        with open(self.labels_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.df = pd.DataFrame(self.data)
        print(f"Loaded {len(self.df)} annotated posts")
        
        # Translation mappings
        self.translations = {
            # Behavior types
            '原创博文': 'Original Post',
            '仅转发': 'Repost Only',
            '转发并评论': 'Repost with Comment',
            '评论': 'Comment',
            '未知': 'Unknown',
            
            # Emotions
            '愤怒': 'Anger',
            '厌恶': 'Disgust',
            '焦虑': 'Anxiety',
            '悲伤': 'Sadness',
            '幸灾乐祸': 'Schadenfreude',
            '兴奋': 'Excitement',
            '中性': 'Neutral',
            
            # Intensity
            '低': 'Low',
            '中等': 'Medium',
            '高': 'High',
            '极高': 'Very High',
            
            # Stance
            '支持': 'Support',
            '反对': 'Oppose',
            '中立': 'Neutral',
            
            # Expression style
            '阴阳怪气': 'Sarcastic',
            '讽刺': 'Ironic',
            '激进': 'Radical',
            '嘲讽': 'Mocking',
            '情绪宣泄': 'Emotional Venting',
            '质疑': 'Questioning',
            '共情': 'Empathetic',
            '冷漠': 'Indifferent',
            '理性': 'Rational',
            '客观': 'Objective',
            
            # Narrative strategy
            '贴标签': 'Labeling',
            '道德绑架': 'Moral Coercion',
            '阴谋论': 'Conspiracy Theory',
            '转移话题': 'Topic Shifting',
            '号召行动': 'Call to Action',
            '人身攻击': 'Personal Attack',
            '质疑事实': 'Fact Questioning',
            '陈述事实': 'Fact Stating',
            '提供证据': 'Evidence Providing',
            
            # Sentiment polarity
            '负面': 'Negative',
            '正面': 'Positive',
            
            # Attitude
            
            # Belief
            '相信此事件': 'Believe Event',
            '怀疑此事件': 'Doubt Event',
            '不确定': 'Uncertain',
            
            # Politeness
            '礼貌': 'Polite',
            '一般': 'Normal',
            '不礼貌': 'Impolite',
            '粗鲁': 'Rude',
            
            # Emotionality
            '中': 'Medium'
        }
    
    def translate(self, text):
        """Translate Chinese to English"""
        return self.translations.get(text, text)
    
    def plot_behavior_type_distribution(self):
        """Plot behavior type distribution"""
        behavior_counts = self.df['behavior_type'].value_counts()
        
        # Translate to English
        translation_map = {
            '长博文': 'Long Post',
            '短博文': 'Short Post',
            '仅转发': 'Repost Only',
            '转发并长评论': 'Repost with Long Comment',
            '转发并短评论': 'Repost with Short Comment',
            '长评论': 'Long Comment',
            '短评论': 'Short Comment',
            '未知': 'Unknown'
        }
        
        behavior_counts.index = [translation_map.get(x, x) for x in behavior_counts.index]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        behavior_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Distribution of Behavior Types', fontsize=14, fontweight='bold')
        ax.set_xlabel('Behavior Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'behavior_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: behavior_type_distribution.png")
    
    def plot_emotion_distribution(self):
        """Plot emotion type distribution"""
        emotions = []
        for _, row in self.df.iterrows():
            llm_ann = row.get('llm_annotation', {})
            if isinstance(llm_ann, dict) and 'emotion_type' in llm_ann:
                emo_list = llm_ann['emotion_type']
                if isinstance(emo_list, list):
                    emotions.extend(emo_list)
        
        if not emotions:
            print("⚠ No emotion data available")
            return
        
        emotion_counts = Counter(emotions)
        emotion_df = pd.DataFrame(emotion_counts.items(), columns=['Emotion', 'Count'])
        emotion_df['Emotion'] = emotion_df['Emotion'].apply(self.translate)
        emotion_df = emotion_df.sort_values('Count', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=emotion_df, x='Emotion', y='Count', ax=ax, palette='viridis')
        ax.set_title('Distribution of Emotion Types', fontsize=14, fontweight='bold')
        ax.set_xlabel('Emotion Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'emotion_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: emotion_distribution.png")
    
    def plot_stance_distribution(self):
        """Plot stance distribution"""
        stances = []
        for _, row in self.df.iterrows():
            llm_ann = row.get('llm_annotation', {})
            if isinstance(llm_ann, dict) and 'stance' in llm_ann:
                stances.append(llm_ann['stance'])
        
        if not stances:
            print("⚠ No stance data available")
            return
        
        stance_counts = Counter(stances)
        stance_df = pd.DataFrame(stance_counts.items(), columns=['Stance', 'Count'])
        stance_df['Stance'] = stance_df['Stance'].apply(self.translate)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
        ax.pie(stance_df['Count'], labels=stance_df['Stance'], autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax.set_title('Stance Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: stance_distribution.png")
    
    def plot_intensity_heatmap(self):
        """Plot emotion and stance intensity heatmap"""
        emotion_intensities = []
        stance_intensities = []
        
        for _, row in self.df.iterrows():
            llm_ann = row.get('llm_annotation', {})
            if isinstance(llm_ann, dict):
                if 'emotion_intensity' in llm_ann:
                    emotion_intensities.append(llm_ann['emotion_intensity'])
                if 'stance_intensity' in llm_ann:
                    stance_intensities.append(llm_ann['stance_intensity'])
        
        if not emotion_intensities or not stance_intensities:
            print("⚠ No intensity data available")
            return
        
        emo_counts = Counter(emotion_intensities)
        stance_counts = Counter(stance_intensities)
        
        intensity_levels = ['低', '中等', '高', '极高']
        intensity_levels_en = [self.translate(x) for x in intensity_levels]
        
        data = []
        for level in intensity_levels:
            data.append([emo_counts.get(level, 0), stance_counts.get(level, 0)])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data, annot=True, fmt='d', cmap='YlOrRd', 
                    xticklabels=['Emotion Intensity', 'Stance Intensity'],
                    yticklabels=intensity_levels_en, ax=ax)
        ax.set_title('Intensity Distribution Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'intensity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: intensity_heatmap.png")
    
    def plot_sentiment_comparison(self):
        """Compare LLM and NLP sentiment analysis"""
        llm_sentiments = []
        nlp_sentiments = []
        
        for _, row in self.df.iterrows():
            llm_ann = row.get('llm_annotation', {})
            nlp_ann = row.get('nlp_sentiment', {})
            
            if isinstance(llm_ann, dict) and 'sentiment_polarity' in llm_ann:
                llm_sentiments.append(llm_ann['sentiment_polarity'])
            
            if isinstance(nlp_ann, dict) and 'polarity' in nlp_ann:
                nlp_sentiments.append(nlp_ann['polarity'])
        
        if not llm_sentiments or not nlp_sentiments:
            print("⚠ No sentiment data available")
            return
        
        llm_counts = Counter(llm_sentiments)
        nlp_counts = Counter(nlp_sentiments)
        
        sentiments = ['负面', '中性', '正面']
        sentiments_en = [self.translate(x) for x in sentiments]
        
        llm_values = [llm_counts.get(s, 0) for s in sentiments]
        nlp_values = [nlp_counts.get(s.lower(), 0) for s in ['negative', 'neutral', 'positive']]
        
        x = np.arange(len(sentiments_en))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, llm_values, width, label='LLM Annotation', color='skyblue')
        ax.bar(x + width/2, nlp_values, width, label='NLP Analysis', color='lightcoral')
        
        ax.set_xlabel('Sentiment Polarity', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Sentiment Analysis Comparison: LLM vs NLP', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sentiments_en)
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sentiment_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: sentiment_comparison.png")
    
    def plot_expression_style(self):
        """Plot expression style distribution"""
        styles = []
        for _, row in self.df.iterrows():
            llm_ann = row.get('llm_annotation', {})
            if isinstance(llm_ann, dict) and 'expression_style' in llm_ann:
                style_list = llm_ann['expression_style']
                if isinstance(style_list, list):
                    styles.extend(style_list)
        
        if not styles:
            print("⚠ No expression style data available")
            return
        
        style_counts = Counter(styles)
        style_df = pd.DataFrame(style_counts.items(), columns=['Style', 'Count'])
        style_df['Style'] = style_df['Style'].apply(self.translate)
        style_df = style_df.sort_values('Count', ascending=True).tail(10)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(style_df['Style'], style_df['Count'], color='teal')
        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel('Expression Style', fontsize=12)
        ax.set_title('Top 10 Expression Styles', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'expression_style.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: expression_style.png")
    
    def plot_narrative_strategy(self):
        """Plot narrative strategy distribution"""
        strategies = []
        for _, row in self.df.iterrows():
            llm_ann = row.get('llm_annotation', {})
            if isinstance(llm_ann, dict) and 'narrative_strategy' in llm_ann:
                strat_list = llm_ann['narrative_strategy']
                if isinstance(strat_list, list):
                    strategies.extend(strat_list)
        
        if not strategies:
            print("⚠ No narrative strategy data available")
            return
        
        strat_counts = Counter(strategies)
        strat_df = pd.DataFrame(strat_counts.items(), columns=['Strategy', 'Count'])
        strat_df['Strategy'] = strat_df['Strategy'].apply(self.translate)
        strat_df = strat_df.sort_values('Count', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=strat_df, x='Strategy', y='Count', ax=ax, palette='rocket')
        ax.set_title('Top 10 Narrative Strategies', fontsize=14, fontweight='bold')
        ax.set_xlabel('Narrative Strategy', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'narrative_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: narrative_strategy.png")
    
    def plot_belief_distribution(self):
        """Plot belief distribution"""
        beliefs = []
        for _, row in self.df.iterrows():
            llm_ann = row.get('llm_annotation', {})
            if isinstance(llm_ann, dict) and 'belief' in llm_ann:
                beliefs.append(llm_ann['belief'])
        
        if not beliefs:
            print("⚠ No belief data available")
            return
        
        belief_counts = Counter(beliefs)
        belief_df = pd.DataFrame(belief_counts.items(), columns=['Belief', 'Count'])
        belief_df['Belief'] = belief_df['Belief'].apply(self.translate)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#3498db', '#e74c3c', '#95a5a6']
        ax.pie(belief_df['Count'], labels=belief_df['Belief'], autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title('Belief Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'belief_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: belief_distribution.png")
    
    def plot_confrontation_metrics(self):
        """Plot confrontation intensity metrics (politeness & emotionality)"""
        politeness = []
        emotionality = []
        
        for _, row in self.df.iterrows():
            llm_ann = row.get('llm_annotation', {})
            if isinstance(llm_ann, dict):
                if 'politeness' in llm_ann:
                    politeness.append(llm_ann['politeness'])
                if 'emotionality' in llm_ann:
                    emotionality.append(llm_ann['emotionality'])
        
        if not politeness or not emotionality:
            print("⚠ No confrontation metrics data available")
            return
        
        pol_counts = Counter(politeness)
        emo_counts = Counter(emotionality)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Politeness
        pol_df = pd.DataFrame(pol_counts.items(), columns=['Level', 'Count'])
        pol_df['Level'] = pol_df['Level'].apply(self.translate)
        pol_df = pol_df.sort_values('Count', ascending=False)
        ax1.bar(pol_df['Level'], pol_df['Count'], color='lightgreen')
        ax1.set_title('Politeness Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Politeness Level', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        
        # Emotionality
        emo_df = pd.DataFrame(emo_counts.items(), columns=['Level', 'Count'])
        emo_df['Level'] = emo_df['Level'].apply(self.translate)
        emo_df = emo_df.sort_values('Count', ascending=False)
        ax2.bar(emo_df['Level'], emo_df['Count'], color='salmon')
        ax2.set_title('Emotionality Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Emotionality Level', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confrontation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: confrontation_metrics.png")
    
    def plot_temporal_trends(self):
        """Plot temporal trends of sentiment"""
        # Parse timestamps
        self.df['timestamp'] = pd.to_datetime(self.df['time'], errors='coerce')
        self.df = self.df.dropna(subset=['timestamp'])
        self.df = self.df.sort_values('timestamp')
        
        # Extract sentiment scores
        sentiment_scores = []
        for _, row in self.df.iterrows():
            nlp_ann = row.get('nlp_sentiment', {})
            if isinstance(nlp_ann, dict) and 'score' in nlp_ann:
                sentiment_scores.append(nlp_ann['score'])
            else:
                sentiment_scores.append(0.5)
        
        self.df['sentiment_score'] = sentiment_scores
        
        # Resample by hour
        hourly = self.df.set_index('timestamp').resample('h')['sentiment_score'].mean()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(hourly.index, hourly.values, linewidth=2, color='purple')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral')
        ax.fill_between(hourly.index, hourly.values, 0.5, alpha=0.3, color='purple')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Average Sentiment Score', fontsize=12)
        ax.set_title('Temporal Sentiment Trend (Hourly Average)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_sentiment_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: temporal_sentiment_trend.png")
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*50)
        print("Generating Visualizations")
        print("="*50 + "\n")
        
        self.plot_behavior_type_distribution()
        self.plot_emotion_distribution()
        self.plot_stance_distribution()
        self.plot_intensity_heatmap()
        self.plot_sentiment_comparison()
        self.plot_expression_style()
        self.plot_narrative_strategy()
        self.plot_belief_distribution()
        self.plot_confrontation_metrics()
        self.plot_temporal_trends()
        
        print("\n" + "="*50)
        print(f"All visualizations saved to: {self.output_dir}")
        print("="*50)


if __name__ == "__main__":
    visualizer = AnnotationVisualizer()
    visualizer.generate_all_visualizations()
