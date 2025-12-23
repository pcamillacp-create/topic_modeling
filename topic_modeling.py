# ==================== 1. IMPORT ====================
import pandas as pd
import numpy as np
import os
import random

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Topic Modelling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Visualizzazione
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go

# Configurazione
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("📦 Libraries imported successfully!\n")

# ==================== 2. CONFIGURAZIONE ====================
CONFIG = {
    'csv_path': 'data/arxiv_papers.csv',
    'text_column': 'summary',
    'sample_size': 2000,           # Documenti da analizzare
    'random_seed': 42,            # Per riproducibilità
    'n_topics': 5,                # Numero di topic
    'topics_per_doc': 3,          # Topic mostrati per documento
    'max_features': 2000,         # Vocabolario massimo
}

print("⚙️ Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# ==================== 3. CAMPIONAMENTO CASUALE ====================
print("\n" + "="*70)
print("🎲 STEP 1: Random Sampling from Dataset")
print("="*70)

def load_random_samples(csv_path, text_column, sample_size, random_seed=42):
    """
    Carica campioni casuali dal CSV con riferimento alle righe originali
    """
    print(f"Loading {sample_size} random documents from {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None, None
    
    try:
        # Prima contiamo le righe totali
        print("   Counting total rows...")
        chunk_size = 10000
        total_rows = 0
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=[text_column]):
            total_rows += len(chunk)
        
        print(f"   Total rows in dataset: {total_rows:,}")
        
        if sample_size > total_rows:
            sample_size = total_rows
            print(f"   Adjusted sample size to: {sample_size}")
        
        # Genera indici casuali unici
        random.seed(random_seed)
        random_indices = random.sample(range(total_rows), sample_size)
        random_indices.sort()  # Ordina per leggibilità
        
        print(f"   Selected random indices: {random_indices[:10]}..." if len(random_indices) > 10 
              else f"   Selected indices: {random_indices}")
        
        # Carica solo le righe selezionate
        samples = []
        original_info = []
        
        current_index = 0
        chunk_counter = 0
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            chunk_start = current_index
            chunk_end = current_index + len(chunk) - 1
            
            # Controlla se qualche indice selezionato è in questo chunk
            indices_in_chunk = [i for i in random_indices if chunk_start <= i <= chunk_end]
            
            if indices_in_chunk:
                for idx in indices_in_chunk:
                    row_idx_in_chunk = idx - chunk_start
                    if row_idx_in_chunk < len(chunk):
                        text = str(chunk.iloc[row_idx_in_chunk][text_column])
                        samples.append(text)
                        original_info.append({
                            'original_index': idx,
                            'chunk_number': chunk_counter,
                            'position_in_chunk': row_idx_in_chunk
                        })
            
            current_index += len(chunk)
            chunk_counter += 1
            
            if len(samples) >= sample_size:
                break
        
        print(f"✅ Successfully loaded {len(samples)} random documents")
        return samples, original_info
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None, None

# Carica dati casuali
documents, doc_info = load_random_samples(
    CONFIG['csv_path'], 
    CONFIG['text_column'], 
    CONFIG['sample_size'],
    CONFIG['random_seed']
)

if not documents:
    exit()

# ==================== 4. PREPROCESSING ====================
print("\n" + "="*70)
print("🔧 STEP 2: Text Preprocessing")
print("="*70)

def preprocess_text(text):
    """Preprocessing base"""
    if not isinstance(text, str):
        return ""
    
    # Minuscolo
    text = text.lower()
    
    # Rimuovi numeri e caratteri speciali
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenizza e rimuovi stopwords
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatizzazione
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

print("Processing documents...")
processed_docs = [preprocess_text(doc) for doc in documents]

print(f"✅ Preprocessing completed")
print(f"   Original documents: {len(documents)}")
print(f"   Average length: {np.mean([len(doc.split()) for doc in processed_docs]):.0f} words")

# ==================== 5. TOPIC MODELLING ====================
print("\n" + "="*70)
print("🎯 STEP 3: Topic Modeling")
print("="*70)

print("Creating document-term matrix...")
vectorizer = CountVectorizer(
    max_features=CONFIG['max_features'],
    stop_words='english',
    min_df=2,
    max_df=0.8
)

doc_term_matrix = vectorizer.fit_transform(processed_docs)
feature_names = vectorizer.get_feature_names_out()

print(f"✅ Matrix created: {doc_term_matrix.shape[0]} docs × {doc_term_matrix.shape[1]} words")

print(f"\nTraining LDA with {CONFIG['n_topics']} topics...")
lda = LatentDirichletAllocation(
    n_components=CONFIG['n_topics'],
    random_state=CONFIG['random_seed'],
    max_iter=20,
    learning_method='online'
)

lda.fit(doc_term_matrix)

# Ottieni distribuzioni
doc_topic_matrix = lda.transform(doc_term_matrix)  # Documenti × Topic
topic_word_matrix = lda.components_               # Topic × Parole

print("✅ Topic modeling completed!")

# ==================== 6. VISUALIZZAZIONI ====================
print("\n" + "="*70)
print("📊 STEP 4: Creating Visualizations")
print("="*70)

# Crea directory per le visualizzazioni
os.makedirs("visualizations", exist_ok=True)

# TOP WORDS PER TOPIC (Bar Chart)
print("\n1. Creating topic words visualization...")

fig, axes = plt.subplots(CONFIG['n_topics'], 1, figsize=(12, 4*CONFIG['n_topics']))

for topic_idx in range(CONFIG['n_topics']):
    # Top 10 parole per topic
    topic = topic_word_matrix[topic_idx]
    top_indices = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    word_scores = [topic[i] for i in top_indices]
    
    # Normalizza i punteggi per visualizzazione
    word_scores_norm = [score/max(word_scores) for score in word_scores]
    
    ax = axes[topic_idx] if CONFIG['n_topics'] > 1 else axes
    bars = ax.barh(top_words, word_scores_norm, color=plt.cm.Set3(topic_idx/CONFIG['n_topics']))
    
    # Aggiungi valori
    for i, (bar, score) in enumerate(zip(bars, word_scores)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{score:.1f}', va='center', fontsize=9)
    
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Relative Importance')
    ax.set_title(f'Topic {topic_idx + 1} - Key Words', fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('visualizations/topic_words.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. DISTRIBUZIONE DOCUMENTI PER TOPIC (Pie Chart)
print("\n2. Creating topic distribution pie chart...")

# Trova topic dominante per ogni documento
dominant_topics = np.argmax(doc_topic_matrix, axis=1)
topic_counts = pd.Series(dominant_topics).value_counts().sort_index()

plt.figure(figsize=(10, 8))
colors = plt.cm.Paired(np.linspace(0, 1, CONFIG['n_topics']))

# Prepara etichette
labels = [f'Topic {i+1}' for i in range(CONFIG['n_topics'])]
sizes = [topic_counts.get(i, 0) for i in range(CONFIG['n_topics'])]
percentages = [f'{(size/sum(sizes)*100):.1f}%' for size in sizes]
labels_with_pct = [f'{label}\n({pct}, {size} docs)' 
                  for label, pct, size in zip(labels, percentages, sizes)]

wedges, texts, autotexts = plt.pie(sizes, labels=labels_with_pct, colors=colors,
                                  autopct='', startangle=90, pctdistance=0.85)

# Migliora l'aspetto
plt.setp(autotexts, size=10, weight="bold")
plt.setp(texts, size=11)

# Aggiungi cerchio centrale per donut chart
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title(f'Document Distribution Across {CONFIG["n_topics"]} Topics\n(Total: {len(documents)} documents)', 
          fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('visualizations/topic_distribution_pie.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. HEATMAP: TOPIC PROBABILITIES (primi 20 documenti)
print("\n3. Creating document-topic heatmap...")

n_docs_show = min(20, len(documents))
doc_topic_subset = doc_topic_matrix[:n_docs_show]

plt.figure(figsize=(14, 10))
sns.heatmap(doc_topic_subset, 
           cmap='YlOrRd',
           annot=True,
           fmt='.2f',
           cbar_kws={'label': 'Topic Probability'},
           xticklabels=[f'Topic {i+1}' for i in range(CONFIG['n_topics'])],
           yticklabels=[f'Doc {doc_info[i]["original_index"]+1}' for i in range(n_docs_show)])

plt.title(f'Topic Probabilities for First {n_docs_show} Documents\n(Document IDs show original CSV row numbers)', 
          fontsize=16, pad=20)
plt.xlabel('Topics', fontsize=14)
plt.ylabel('Document (Original Row Number)', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/document_topic_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. INTERACTIVE VISUALIZATION (HTML)
print("\n4. Creating interactive visualization (HTML)...")

try:
    import plotly.graph_objects as go
    
    # Prepara dati per grafico interattivo
    topics = [f'Topic {i+1}' for i in range(CONFIG['n_topics'])]
    
    # Crea grafico a barre orizzontali interattivo
    fig = go.Figure()
    
    for doc_idx in range(min(10, len(documents))):
        topic_probs = doc_topic_matrix[doc_idx]
        fig.add_trace(go.Bar(
            y=topics,
            x=topic_probs,
            name=f'Doc {doc_info[doc_idx]["original_index"]+1}',
            orientation='h',
            text=[f'{p:.1%}' for p in topic_probs],
            textposition='auto',
        ))
    
    fig.update_layout(
        title=f'Topic Distribution for First 10 Documents<br><sup>Document IDs: {[doc_info[i]["original_index"]+1 for i in range(min(10, len(documents)))]}</sup>',
        barmode='group',
        xaxis_title='Topic Probability',
        yaxis_title='Topics',
        height=600,
        showlegend=True,
        legend_title='Document (Original Row)'
    )
    
    fig.write_html('visualizations/interactive_topic_dist.html')
    print("   ✅ Interactive HTML saved: visualizations/interactive_topic_dist.html")
    
except ImportError:
    print("   ⚠️  Plotly not installed. Skipping interactive visualization.")
    print("   Install with: pip install plotly")

# ==================== 7. RISULTATI DETTAGLIATI ====================
print("\n" + "="*70)
print("📝 STEP 5: Detailed Results with Document References")
print("="*70)

print(f"\n🔍 ANALYSIS OF RANDOMLY SAMPLED DOCUMENTS:")
print(f"   Total analyzed: {len(documents)} documents")
print(f"   Random seed: {CONFIG['random_seed']}")
print(f"   Topics discovered: {CONFIG['n_topics']}")

print(f"\n📊 DOCUMENT REFERENCES (Original CSV positions):")
print("-" * 80)
print(f"{'Report ID':<10} {'Original Row':<15} {'Dominant Topic':<15} {'Confidence':<12} {'Preview'}")
print("-" * 80)

for i in range(min(15, len(documents))):
    # Calcola topic dominante
    topic_probs = doc_topic_matrix[i]
    dominant_topic = np.argmax(topic_probs) + 1
    confidence = topic_probs.max() * 100
    
    # Ottieni info documento originale
    orig_idx = doc_info[i]['original_index']
    
    # Anteprima testo
    preview = documents[i]
    if len(preview) > 60:
        preview = preview[:57] + "..."
    
    print(f"Doc-{i+1:<8} {orig_idx+1:<14} Topic {dominant_topic:<12} {confidence:>6.1f}%   {preview}")

# Mostra alcuni esempi dettagliati
print(f"\n🎯 DETAILED EXAMPLES (First 3 documents):")
print("="*80)

for i in range(min(3, len(documents))):
    print(f"\n📄 DOCUMENT {i+1} (Original CSV row: {doc_info[i]['original_index'] + 1}):")
    print("-" * 60)
    
    # Testo originale
    print(f"ORIGINAL ABSTRACT:")
    print(f"{documents[i][:200]}..." if len(documents[i]) > 200 else documents[i])
    
    # Distribuzione topic
    topic_probs = doc_topic_matrix[i]
    sorted_indices = topic_probs.argsort()[::-1]  # Ordina discendente
    
    print(f"\nTOPIC DISTRIBUTION (normalized to 100%):")
    print("-" * 40)
    
    total_prob = topic_probs.sum()
    for rank, topic_idx in enumerate(sorted_indices[:CONFIG['topics_per_doc']], 1):
        prob = topic_probs[topic_idx] / total_prob  # Normalizza
        prob_percent = prob * 100
        
        # Top parole per questo topic
        topic = topic_word_matrix[topic_idx]
        top_word_indices = topic.argsort()[-5:][::-1]
        top_words = [feature_names[idx] for idx in top_word_indices]
        
        print(f"{rank}. Topic {topic_idx + 1}: {prob_percent:5.1f}%")
        print(f"   Key words: {', '.join(top_words)}")
    
    print("-" * 60)

# ==================== 8. SALVATAGGIO RISULTATI COMPLETI ====================
print("\n" + "="*70)
print("💾 STEP 6: Saving Complete Results")
print("="*70)

# Crea DataFrame con tutti i risultati
results_data = []

for i in range(len(documents)):
    topic_probs = doc_topic_matrix[i]
    total_prob = topic_probs.sum()
    
    # Normalizza le probabilità
    topic_probs_norm = topic_probs / total_prob if total_prob > 0 else topic_probs
    
    # Trova topic dominanti
    sorted_indices = topic_probs_norm.argsort()[::-1]
    
    row_data = {
        'report_id': f"Doc-{i+1}",
        'original_csv_row': doc_info[i]['original_index'] + 1,  # +1 per base 1
        'chunk_number': doc_info[i]['chunk_number'],
        'position_in_chunk': doc_info[i]['position_in_chunk'],
        'original_text': documents[i],
        'processed_text': processed_docs[i],
        'word_count': len(processed_docs[i].split())
    }
    
    # Aggiungi probabilità topic
    for topic_idx in range(CONFIG['n_topics']):
        row_data[f'topic_{topic_idx+1}_prob'] = float(topic_probs_norm[topic_idx])
        row_data[f'topic_{topic_idx+1}_percent'] = float(topic_probs_norm[topic_idx] * 100)
    
    # Aggiungi info topic dominanti
    for rank in range(min(3, CONFIG['n_topics'])):
        if rank < len(sorted_indices):
            top_topic_idx = sorted_indices[rank]
            row_data[f'top_{rank+1}_topic'] = int(top_topic_idx + 1)
            row_data[f'top_{rank+1}_confidence'] = float(topic_probs_norm[top_topic_idx] * 100)
        else:
            row_data[f'top_{rank+1}_topic'] = None
            row_data[f'top_{rank+1}_confidence'] = None
    
    results_data.append(row_data)

results_df = pd.DataFrame(results_data)
results_df.to_csv('visualizations/complete_analysis_results.csv', index=False, encoding='utf-8-sig')

print("✅ Complete results saved: visualizations/complete_analysis_results.csv")

# Salva anche le parole per topic
topic_words_data = []
for topic_idx in range(CONFIG['n_topics']):
    topic = topic_word_matrix[topic_idx]
    top_indices = topic.argsort()[-20:][::-1]
    
    for rank, word_idx in enumerate(top_indices, 1):
        word = feature_names[word_idx]
        score = topic[word_idx]
        
        topic_words_data.append({
            'topic': topic_idx + 1,
            'word': word,
            'score': float(score),
            'rank': rank
        })

topic_words_df = pd.DataFrame(topic_words_data)
topic_words_df.to_csv('visualizations/topic_words_details.csv', index=False)

print("✅ Topic words saved: visualizations/topic_words_details.csv")

# ==================== 9. REPORT FINALE ====================
print("\n" + "="*70)
print("🏆 ANALYSIS COMPLETED!")
print("="*70)

print(f"\n📁 FILES CREATED in 'visualizations/' folder:")
print("   1. topic_words.png - Key words for each topic")
print("   2. topic_distribution_pie.png - Document distribution")
print("   3. document_topic_heatmap.png - Topic probabilities heatmap")
print("   4. interactive_topic_dist.html - Interactive topic distribution")
print("   5. complete_analysis_results.csv - Full analysis data")
print("   6. topic_words_details.csv - Word scores per topic")

print(f"\n🔍 HOW TO INTERPRET:")
print("   • Original CSV row numbers are preserved for reference")
print("   • Each 'Doc-X' corresponds to a specific row in your CSV")
print("   • You can trace back any result to the original abstract")

print(f"\n🎲 SAMPLING METHOD:")
print(f"   • Random sampling with seed {CONFIG['random_seed']}")
print(f"   • {CONFIG['sample_size']} documents from {CONFIG['csv_path']}")
print(f"   • Original positions preserved in 'original_csv_row' column")

print(f"\n📊 NEXT STEPS:")
print("   1. Open complete_analysis_results.csv to see all results")
print("   2. Check the PNG files for visual insights")
print("   3. Use original_csv_row to find specific abstracts in your dataset")

print("\n" + "="*70)
print("✅ Random sampling with visualizations completed!")
print("="*70)

# Mostra collegamento tra ID report e righe originali
print(f"\n📋 QUICK REFERENCE - First 10 documents:")
print("-" * 50)
print(f"{'Report ID':<10} {'CSV Row':<10}")
print("-" * 50)
for i in range(min(10, len(doc_info))):
    print(f"Doc-{i+1:<8} {doc_info[i]['original_index'] + 1:<10}")