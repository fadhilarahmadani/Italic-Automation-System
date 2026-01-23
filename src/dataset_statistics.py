# src/dataset_statistics.py
"""
Dataset statistics and analysis for italic detection

This module provides comprehensive statistics about the dataset to understand
data distribution and quality.
"""
import json
from typing import List, Dict, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import config
from utils import setup_logging

logger = setup_logging(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class DatasetStatistics:
    """Analyze dataset statistics"""

    def __init__(self, dataset_path: Path):
        """
        Initialize with dataset

        Args:
            dataset_path: Path to dataset JSON file
        """
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} samples")

    def compute_basic_stats(self) -> Dict:
        """Compute basic dataset statistics"""
        total_samples = len(self.data)
        total_tokens = sum(len(item['tokens']) for item in self.data)
        total_italic_words = sum(
            sum(1 for label in item['labels'] if label == 'B')
            for item in self.data
        )

        # Label distribution
        all_labels = []
        for item in self.data:
            all_labels.extend(item['labels'])

        label_counts = Counter(all_labels)

        # Sentence length distribution
        sentence_lengths = [len(item['tokens']) for item in self.data]

        # Italic phrase length distribution
        italic_phrase_lengths = []
        for item in self.data:
            current_phrase_len = 0
            for label in item['labels']:
                if label == 'B':
                    if current_phrase_len > 0:
                        italic_phrase_lengths.append(current_phrase_len)
                    current_phrase_len = 1
                elif label == 'I':
                    current_phrase_len += 1
                else:  # 'O'
                    if current_phrase_len > 0:
                        italic_phrase_lengths.append(current_phrase_len)
                    current_phrase_len = 0

            if current_phrase_len > 0:
                italic_phrase_lengths.append(current_phrase_len)

        stats = {
            'total_samples': total_samples,
            'total_tokens': total_tokens,
            'total_italic_phrases': total_italic_words,
            'avg_tokens_per_sample': total_tokens / total_samples,
            'avg_italic_per_sample': total_italic_words / total_samples,
            'label_distribution': {
                'O': label_counts['O'],
                'B': label_counts['B'],
                'I': label_counts['I'],
                'O_percentage': label_counts['O'] / total_tokens * 100,
                'B_percentage': label_counts['B'] / total_tokens * 100,
                'I_percentage': label_counts['I'] / total_tokens * 100,
            },
            'sentence_length_stats': {
                'min': min(sentence_lengths),
                'max': max(sentence_lengths),
                'mean': sum(sentence_lengths) / len(sentence_lengths),
                'median': sorted(sentence_lengths)[len(sentence_lengths) // 2],
            },
            'italic_phrase_length_stats': {
                'min': min(italic_phrase_lengths) if italic_phrase_lengths else 0,
                'max': max(italic_phrase_lengths) if italic_phrase_lengths else 0,
                'mean': sum(italic_phrase_lengths) / len(italic_phrase_lengths) if italic_phrase_lengths else 0,
                'median': sorted(italic_phrase_lengths)[len(italic_phrase_lengths) // 2] if italic_phrase_lengths else 0,
                'distribution': Counter(italic_phrase_lengths),
            },
        }

        return stats

    def extract_italic_phrases(self) -> List[str]:
        """Extract all italic phrases from dataset"""
        phrases = []

        for item in self.data:
            tokens = item['tokens']
            labels = item['labels']

            current_phrase = []
            for token, label in zip(tokens, labels):
                if label == 'B':
                    if current_phrase:
                        phrases.append(' '.join(current_phrase))
                    current_phrase = [token]
                elif label == 'I':
                    current_phrase.append(token)
                else:  # 'O'
                    if current_phrase:
                        phrases.append(' '.join(current_phrase))
                    current_phrase = []

            if current_phrase:
                phrases.append(' '.join(current_phrase))

        return phrases

    def analyze_phrase_patterns(self) -> Dict:
        """Analyze patterns in italic phrases"""
        phrases = self.extract_italic_phrases()

        # Frequency analysis
        phrase_freq = Counter(phrases)

        # Word count distribution
        word_counts = Counter(len(phrase.split()) for phrase in phrases)

        # Character patterns
        contains_hyphen = sum(1 for p in phrases if '-' in p)
        contains_underscore = sum(1 for p in phrases if '_' in p)
        contains_numbers = sum(1 for p in phrases if any(c.isdigit() for c in p))
        all_caps = sum(1 for p in phrases if p.isupper())
        title_case = sum(1 for p in phrases if p.istitle())

        # Language patterns (simple heuristic)
        likely_english = sum(
            1 for p in phrases
            if any(word in p.lower() for word in ['the', 'and', 'of', 'to', 'in'])
        )

        analysis = {
            'total_unique_phrases': len(phrase_freq),
            'total_phrase_occurrences': len(phrases),
            'most_common_phrases': dict(phrase_freq.most_common(50)),
            'word_count_distribution': dict(word_counts),
            'pattern_statistics': {
                'contains_hyphen': contains_hyphen,
                'contains_hyphen_pct': contains_hyphen / len(phrases) * 100,
                'contains_underscore': contains_underscore,
                'contains_underscore_pct': contains_underscore / len(phrases) * 100,
                'contains_numbers': contains_numbers,
                'contains_numbers_pct': contains_numbers / len(phrases) * 100,
                'all_caps': all_caps,
                'all_caps_pct': all_caps / len(phrases) * 100,
                'title_case': title_case,
                'title_case_pct': title_case / len(phrases) * 100,
                'likely_english': likely_english,
                'likely_english_pct': likely_english / len(phrases) * 100,
            }
        }

        return analysis

    def analyze_context(self) -> Dict:
        """Analyze context around italic phrases"""
        context_patterns = defaultdict(int)

        for item in self.data:
            tokens = item['tokens']
            labels = item['labels']

            for i, label in enumerate(labels):
                if label == 'B':
                    # Analyze preceding word
                    if i > 0:
                        prev_token = tokens[i-1].lower()
                        context_patterns[f'before_{prev_token}'] += 1

                    # Find end of phrase
                    j = i + 1
                    while j < len(labels) and labels[j] == 'I':
                        j += 1

                    # Analyze following word
                    if j < len(tokens):
                        next_token = tokens[j].lower()
                        context_patterns[f'after_{next_token}'] += 1

        # Get most common context words
        before_words = {
            k.replace('before_', ''): v
            for k, v in context_patterns.items()
            if k.startswith('before_')
        }
        after_words = {
            k.replace('after_', ''): v
            for k, v in context_patterns.items()
            if k.startswith('after_')
        }

        return {
            'most_common_before': dict(
                Counter(before_words).most_common(20)
            ),
            'most_common_after': dict(
                Counter(after_words).most_common(20)
            ),
        }

    def generate_visualizations(self, output_dir: Path):
        """Generate visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating visualizations...")

        # 1. Label distribution pie chart
        stats = self.compute_basic_stats()
        labels = ['O (Non-italic)', 'B (Begin)', 'I (Inside)']
        sizes = [
            stats['label_distribution']['O'],
            stats['label_distribution']['B'],
            stats['label_distribution']['I'],
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.1, 0, 0)

        plt.figure(figsize=(10, 7))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title('Label Distribution in Dataset', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.savefig(output_dir / 'label_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: label_distribution.png")

        # 2. Sentence length distribution
        sentence_lengths = [len(item['tokens']) for item in self.data]
        plt.figure(figsize=(12, 6))
        plt.hist(sentence_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Tokens', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Sentence Length Distribution', fontsize=16, fontweight='bold')
        plt.axvline(stats['sentence_length_stats']['mean'], color='red',
                   linestyle='--', label=f"Mean: {stats['sentence_length_stats']['mean']:.1f}")
        plt.axvline(stats['sentence_length_stats']['median'], color='green',
                   linestyle='--', label=f"Median: {stats['sentence_length_stats']['median']}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'sentence_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: sentence_length_distribution.png")

        # 3. Italic phrase length distribution
        phrase_analysis = self.analyze_phrase_patterns()
        word_counts = phrase_analysis['word_count_distribution']

        plt.figure(figsize=(12, 6))
        x = sorted(word_counts.keys())
        y = [word_counts[k] for k in x]
        plt.bar(x, y, color='coral', edgecolor='black', alpha=0.7)
        plt.xlabel('Phrase Length (words)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Italic Phrase Length Distribution', fontsize=16, fontweight='bold')
        plt.xticks(x)
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(output_dir / 'phrase_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: phrase_length_distribution.png")

        # 4. Most common italic phrases (top 20)
        most_common = list(phrase_analysis['most_common_phrases'].items())[:20]
        phrases, counts = zip(*most_common)

        plt.figure(figsize=(14, 8))
        y_pos = range(len(phrases))
        plt.barh(y_pos, counts, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.yticks(y_pos, phrases, fontsize=10)
        plt.xlabel('Frequency', fontsize=12)
        plt.title('Top 20 Most Common Italic Phrases', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / 'most_common_phrases.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: most_common_phrases.png")

        # 5. Pattern statistics (bar chart)
        pattern_stats = phrase_analysis['pattern_statistics']
        patterns = ['Hyphen', 'Underscore', 'Numbers', 'All Caps', 'Title Case', 'English']
        percentages = [
            pattern_stats['contains_hyphen_pct'],
            pattern_stats['contains_underscore_pct'],
            pattern_stats['contains_numbers_pct'],
            pattern_stats['all_caps_pct'],
            pattern_stats['title_case_pct'],
            pattern_stats['likely_english_pct'],
        ]

        plt.figure(figsize=(12, 6))
        plt.bar(patterns, percentages, color='mediumpurple', edgecolor='black', alpha=0.7)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.title('Italic Phrase Pattern Analysis', fontsize=16, fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(percentages):
            plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

        plt.savefig(output_dir / 'pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: pattern_analysis.png")

        logger.info(f"\nAll visualizations saved to {output_dir}/")

    def generate_report(self, output_path: Path):
        """Generate comprehensive text report"""
        logger.info("Generating statistics report...")

        stats = self.compute_basic_stats()
        phrase_analysis = self.analyze_phrase_patterns()
        context_analysis = self.analyze_context()

        lines = []
        lines.append("=" * 80)
        lines.append("DATASET STATISTICS REPORT")
        lines.append("Italic Detection Dataset")
        lines.append("=" * 80)
        lines.append("")

        # Basic statistics
        lines.append("BASIC STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Total Samples:          {stats['total_samples']:,}")
        lines.append(f"Total Tokens:           {stats['total_tokens']:,}")
        lines.append(f"Total Italic Phrases:   {stats['total_italic_phrases']:,}")
        lines.append(f"Avg Tokens/Sample:      {stats['avg_tokens_per_sample']:.2f}")
        lines.append(f"Avg Italic/Sample:      {stats['avg_italic_per_sample']:.2f}")
        lines.append("")

        # Label distribution
        lines.append("LABEL DISTRIBUTION")
        lines.append("-" * 80)
        lines.append(f"O (Non-italic):  {stats['label_distribution']['O']:,} ({stats['label_distribution']['O_percentage']:.2f}%)")
        lines.append(f"B (Begin):       {stats['label_distribution']['B']:,} ({stats['label_distribution']['B_percentage']:.2f}%)")
        lines.append(f"I (Inside):      {stats['label_distribution']['I']:,} ({stats['label_distribution']['I_percentage']:.2f}%)")
        lines.append("")

        # Sentence length statistics
        lines.append("SENTENCE LENGTH STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Minimum:   {stats['sentence_length_stats']['min']}")
        lines.append(f"Maximum:   {stats['sentence_length_stats']['max']}")
        lines.append(f"Mean:      {stats['sentence_length_stats']['mean']:.2f}")
        lines.append(f"Median:    {stats['sentence_length_stats']['median']}")
        lines.append("")

        # Phrase analysis
        lines.append("ITALIC PHRASE ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Unique Phrases:         {phrase_analysis['total_unique_phrases']:,}")
        lines.append(f"Total Occurrences:      {phrase_analysis['total_phrase_occurrences']:,}")
        lines.append(f"Avg Repetition:         {phrase_analysis['total_phrase_occurrences'] / phrase_analysis['total_unique_phrases']:.2f}")
        lines.append("")

        # Pattern statistics
        lines.append("PATTERN STATISTICS")
        lines.append("-" * 80)
        pattern_stats = phrase_analysis['pattern_statistics']
        lines.append(f"Contains Hyphen:     {pattern_stats['contains_hyphen']:,} ({pattern_stats['contains_hyphen_pct']:.1f}%)")
        lines.append(f"Contains Underscore: {pattern_stats['contains_underscore']:,} ({pattern_stats['contains_underscore_pct']:.1f}%)")
        lines.append(f"Contains Numbers:    {pattern_stats['contains_numbers']:,} ({pattern_stats['contains_numbers_pct']:.1f}%)")
        lines.append(f"All Caps:            {pattern_stats['all_caps']:,} ({pattern_stats['all_caps_pct']:.1f}%)")
        lines.append(f"Title Case:          {pattern_stats['title_case']:,} ({pattern_stats['title_case_pct']:.1f}%)")
        lines.append(f"Likely English:      {pattern_stats['likely_english']:,} ({pattern_stats['likely_english_pct']:.1f}%)")
        lines.append("")

        # Most common phrases
        lines.append("MOST COMMON ITALIC PHRASES (Top 20)")
        lines.append("-" * 80)
        for i, (phrase, count) in enumerate(list(phrase_analysis['most_common_phrases'].items())[:20], 1):
            lines.append(f"{i:2d}. {phrase:50s}  {count:4d}")
        lines.append("")

        # Context analysis
        lines.append("MOST COMMON CONTEXT WORDS")
        lines.append("-" * 80)
        lines.append("Words appearing BEFORE italic phrases:")
        for word, count in list(context_analysis['most_common_before'].items())[:10]:
            lines.append(f"  {word:20s}  {count:4d}")
        lines.append("")
        lines.append("Words appearing AFTER italic phrases:")
        for word, count in list(context_analysis['most_common_after'].items())[:10]:
            lines.append(f"  {word:20s}  {count:4d}")
        lines.append("")

        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"Report saved to {output_path}")


def main():
    """Run dataset statistics analysis"""
    logger.info("="*60)
    logger.info("Dataset Statistics Analysis")
    logger.info("="*60)

    # Analyze each split
    splits = ['train', 'validation', 'test', 'italic_dataset']

    for split in splits:
        dataset_path = config.DATA_DIR / f"{split}.json"
        if not dataset_path.exists():
            logger.warning(f"Skipping {split}: file not found")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {split.upper()} dataset")
        logger.info(f"{'='*60}")

        # Analyze
        analyzer = DatasetStatistics(dataset_path)

        # Generate report
        report_path = config.MODEL_DIR / f"{split}_statistics.txt"
        analyzer.generate_report(report_path)

        # Generate visualizations
        viz_dir = config.MODEL_DIR / f"{split}_visualizations"
        analyzer.generate_visualizations(viz_dir)

        # Save JSON stats
        stats = analyzer.compute_basic_stats()
        phrase_analysis = analyzer.analyze_phrase_patterns()
        context_analysis = analyzer.analyze_context()

        json_path = config.MODEL_DIR / f"{split}_statistics.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'basic_stats': stats,
                'phrase_analysis': phrase_analysis,
                'context_analysis': context_analysis,
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON stats saved to {json_path}")


if __name__ == "__main__":
    main()
