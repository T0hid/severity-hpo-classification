"""
Figures 2 to 6 and Supplementary Figures S1 to S3, the validation and analysis reports, and the
copies of Figure 7 under their paper names.

Run through the entry point (python run.py figures or python run.py verify). Reads the agent
classification file, the ground truth (data/ground_truth.xlsx), the model-selection summary
(published_run/model_selection_summary.csv), the baseline scores per term and the Figure 7 outputs.
Writes, under the output folder:
    main/              Figures 2 to 7 (PDF and PNG)
    supplementary/     Figures S1 to S3 (PDF and PNG)
    data_and_reports/  validation and analysis reports, scored terms, Figure 2E scores, Figure 7 data
    _work/             the raw outputs before renaming

Parts:
  Part 1  Supplementary Figure S1 (model selection).
  Part 2  Figures 2 to 6, S2 and S3 (validation against the ground truth, agent vs baseline, and the
          phenotype-level analysis of all classified terms).
  Part 4  copies every figure to its paper name; Figure 7 is drawn by gene_level.py.
Figure 1 (workflow) and Supplementary Figure S4 (worked example) are drawings, not code output.
"""
import matplotlib


def run(output_dir, baseline_per_term, gene_level_dir, classifications=None):
    """Draws the figures and writes the reports to output_dir."""
    from hpo_severity import style; style.reset()   # the matplotlib settings of the paper figures
    import matplotlib.pyplot as _plt; _plt.close("all")
    matplotlib.use('Agg')
    import shutil
    from pathlib import Path

    from . import paths
    ROOT = paths.ROOT
    CLASSIFICATIONS = Path(classifications or paths.CLASSIFICATIONS)
    GT_XLSX = paths.GROUND_TRUTH
    MODEL_SELECTION_CSV = paths.MODEL_SELECTION_SUMMARY
    BASELINE_PER_TERM = Path(baseline_per_term)
    FIG7_DIR = Path(gene_level_dir)
    OUTPUT = Path(output_dir)
    WORK = OUTPUT / '_work'
    for _d in ('S1', 'notebook'):
        (WORK / _d).mkdir(parents=True, exist_ok=True)


    def reset_style():
        import matplotlib.pyplot as _plt
        _plt.close('all')
        from hpo_severity import style; style.reset()
        try:
            import seaborn as _sns
            _sns.reset_orig()
        except ImportError:
            pass




    # ####################################################################################################
    # PART 1: Supplementary Figure S1, model selection
    # ####################################################################################################
    import os
    import glob
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import matplotlib.patheffects as path_effects
    from datetime import datetime

    # ==============================================================================
    # SCRIPT CONFIGURATION
    # ==============================================================================
    # folder of the model-selection summary
    RESULTS_INPUT_DIR = str(MODEL_SELECTION_CSV.parent)

    # Global font size and style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['figure.titlesize'] = 26
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['legend.title_fontsize'] = 20
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 20 # Increased model name font size on axes
    SAVE_DPI = 300

    # ==============================================================================
    # STEP 1: SETUP OUTPUT DIRECTORY
    # ==============================================================================
    print("Step 1: Setting up output directory")
    # Create a new sub-directory with a timestamp to store the plots
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    PLOTS_OUTPUT_DIR = str(WORK / 'S1')   # path

    try:
        os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
        print(f"Plots will be saved in: {PLOTS_OUTPUT_DIR}")
    except OSError as e:
        print(f"ERROR: Could not create directory {PLOTS_OUTPUT_DIR}. Error: {e}")
        exit()

    # ==============================================================================
    # STEP 2: AUTOMATICALLY FIND AND LOAD THE LATEST RESULTS FILE
    # ==============================================================================
    print("\nStep 2: Finding latest results file")
    search_pattern = str(MODEL_SELECTION_CSV)   # the model-selection summary of the paper's run
    summary_files = glob.glob(search_pattern)

    if not summary_files:
        print(f"ERROR: No summary CSV files found in the directory: {RESULTS_INPUT_DIR}")
        exit()

    latest_file = max(summary_files, key=os.path.getctime)
    print(f"Loading data from: {os.path.basename(latest_file)}")
    df = pd.read_csv(latest_file)

    # Clean up model names for prettier labels
    df['model_short_name'] = df['model'].apply(lambda x: x.split('/')[-1])

    # ==============================================================================
    # STEP 3: PERFORM DATA ANALYSIS
    # ==============================================================================
    print("\nStep 3: Analyzing data")
    # Find the trial with the best accuracy for each model
    best_per_model_accuracy = df.loc[df.groupby('model_short_name')['overall_accuracy'].idxmax()]

    # Calculate an efficiency score (accuracy per second of response time)
    df['efficiency_score'] = df['overall_accuracy'] / df['avg_response_time']
    best_efficient_per_model = df.loc[df.groupby('model_short_name')['efficiency_score'].idxmax()].sort_values('efficiency_score', ascending=False)
    print("Data analysis complete.")

    # ==============================================================================
    # STEP 4: GENERATE ENHANCED VISUALIZATIONS
    # ==============================================================================
    print("\nStep 4: Generating visualizations")
    # Define a consistent color palette for models
    model_palette = sns.color_palette("viridis", n_colors=df['model_short_name'].nunique())
    model_color_map = dict(zip(df['model_short_name'].unique(), model_palette))

    # Visualization 1: Best Accuracy per Model (Bar Chart)
    plt.figure(figsize=(16, 9))
    bar_plot = sns.barplot(
        x='overall_accuracy',
        y='model_short_name',
        data=best_per_model_accuracy.sort_values('overall_accuracy', ascending=False),
        palette=model_palette,
        hue='model_short_name',
        dodge=False
    )
    plt.title('Peak Overall Accuracy by Model')
    plt.xlabel('Best Overall Accuracy')
    plt.ylabel('Model')
    plt.xlim(0, 1)
    plt.legend([],[], frameon=False) # Hide the redundant legend

    # Add clear labels to each bar
    for i, (index, row) in enumerate(best_per_model_accuracy.sort_values('overall_accuracy', ascending=False).iterrows()):
        text = plt.text(row.overall_accuracy + 0.01, i, f'{row.overall_accuracy:.2%}',
                        va='center', ha='left', fontsize=18, fontweight='bold', color='black')
        text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

    plt.tight_layout()
    accuracy_plot_path = os.path.join(PLOTS_OUTPUT_DIR, '1_best_accuracy_per_model.png')
    style.save(plt.gcf(), accuracy_plot_path)
    print(f"Saved Best Accuracy plot: {os.path.basename(accuracy_plot_path)}")
    plt.close()


    # Visualization 2: Consolidated Performance View (Scatter Plot)
    plt.figure(figsize=(20, 12))
    scatter_plot = sns.scatterplot(
        data=df,
        x='avg_response_time',
        y='overall_accuracy',
        hue='temperature',
        style='model_short_name',
        size='top_p',
        sizes=(50, 500),
        palette='magma',
        s=200 # base size for markers
    )

    # Annotate the best point for each model to reduce clutter
    for idx, row in best_per_model_accuracy.iterrows():
        text = plt.text(row['avg_response_time'], row['overall_accuracy'] + 0.01,
                        f"Best {row['model_short_name']}\n{row['overall_accuracy']:.2%}",
                        fontdict={'ha': 'center', 'size': 16, 'weight': 'bold'}) # Increased annotation font size
        text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

    plt.title('Consolidated Model Performance: Accuracy vs. Time', fontsize=28, pad=20)
    plt.xlabel('Average Response Time (seconds)')
    plt.ylabel('Overall Accuracy')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    tradeoff_plot_path = os.path.join(PLOTS_OUTPUT_DIR, '2_consolidated_performance_view.png')
    style.save(plt.gcf(), tradeoff_plot_path)
    print(f"Saved Consolidated Performance plot: {os.path.basename(tradeoff_plot_path)}")
    plt.close()


    # Visualization 3: Combined Performance Heatmap (Accuracy & Time)
    # Pivot for accuracy (used for color)
    accuracy_pivot = df.pivot_table(
        index='model_short_name', columns='temperature', values='overall_accuracy'
    )
    # Pivot for time (used for text annotation)
    time_pivot = df.pivot_table(
        index='model_short_name', columns='temperature', values='avg_response_time'
    )

    # Create custom string annotations
    annot_labels = np.full_like(accuracy_pivot, "", dtype=object)
    for i, model in enumerate(accuracy_pivot.index):
        for j, temp in enumerate(accuracy_pivot.columns):
            acc = accuracy_pivot.iat[i, j]
            time = time_pivot.iat[i, j]
            if pd.notna(acc):
                annot_labels[i, j] = f"{acc:.1%}\n({time:.1f}s)"
            else:
                annot_labels[i, j] = "N/A"

    plt.figure(figsize=(20, 12))
    sns.heatmap(
        accuracy_pivot,
        annot=annot_labels,
        fmt="",  # Use empty fmt as we provide custom annotations
        linewidths=.5,
        cmap='cividis',
        annot_kws={"size": 14, "fontweight": "bold", "va": "center"}
    )
    plt.title('Combined Performance: Accuracy (Color) & Response Time (Text)', fontsize=26, pad=20)
    plt.xlabel('Temperature Setting')
    plt.ylabel('Model')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    heatmap_plot_path = os.path.join(PLOTS_OUTPUT_DIR, '3_combined_performance_heatmap.pdf')
    style.save(plt.gcf(), heatmap_plot_path)
    style.save(plt.gcf(), heatmap_plot_path.replace('.pdf', '.png'))   # PNG too
    print(f"Saved Combined Performance Heatmap: {os.path.basename(heatmap_plot_path)}")
    plt.close()


    # Visualization 4: Model Efficiency Score (Bar Chart)
    plt.figure(figsize=(16, 9))
    sns.barplot(
        x='efficiency_score',
        y='model_short_name',
        data=best_efficient_per_model,
        palette=model_palette,
        hue='model_short_name',
        dodge=False
    )
    plt.title('Model Efficiency (Accuracy per Second)')
    plt.xlabel('Efficiency Score (Higher is Better)')
    plt.ylabel('Model')
    plt.legend([],[], frameon=False) # Hide legend

    for i, (index, row) in enumerate(best_efficient_per_model.iterrows()):
        text = plt.text(row.efficiency_score, i, f' {row.efficiency_score:.3f} ',
                        va='center', ha='left', fontsize=18, fontweight='bold', color='black')
        text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

    plt.tight_layout()
    efficiency_plot_path = os.path.join(PLOTS_OUTPUT_DIR, '4_model_efficiency_score.pdf')
    style.save(plt.gcf(), efficiency_plot_path)
    style.save(plt.gcf(), efficiency_plot_path.replace('.pdf', '.png'))   # PNG too
    print(f"Saved Efficiency Score plot: {os.path.basename(efficiency_plot_path)}")
    plt.close()

    print("\nAll tasks complete. Visualizations are saved in the output directory.")
    reset_style()


    # ####################################################################################################
    # PART 2: Figures 2 to 6, S2, S3
    # ####################################################################################################
    """
Combined analysis and validation pipeline.
  - Inputs: the agent classification file (all 10,652 terms) and the ground-truth file (the 898 scored terms,
    also used for the PSI). The agent output is deduplicated by hpo_id.
  - Category names are normalised the same way on both sides.
  - Validation is scored on every matched term (898), including NFC and Out of Scope. For the tier panels NFC is
    placed after Tier 4 (shown as "NFC"); Clinical Sign or Laboratory Abnormality is scored as non-tiered (NFC).

Combined Analysis & Validation Pipeline
========================================
Inputs:
  - CLASSIFICATIONS_CSV  (agent classification of every HPO term)
  - GT_file.xlsx         (ground-truth labels)

Outputs (all written to OUTPUT_DIR):
  Fig3_Clinical_Landscape_and_Onset.pdf
  Fig4_Evidence_Quality.pdf
  Fig5_Tier_and_QoL_Overview.pdf
  FigS2_QoL_Impact_Analysis.pdf
  Fig6_Management_Drivers.pdf
  FigS3CD_Distribution_and_Efficiency.pdf
  analysis_report.txt
  Fig2_validation_composite.pdf
  validation_report.txt
"""

    # =====================================================================
    #  IMPORTS
    # =====================================================================
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from scipy import stats
    from scipy.stats import (
        spearmanr, pearsonr, chi2_contingency,
        kruskal, mannwhitneyu, kendalltau, linregress
    )
    from hpo_severity import exact_p as XP   # exact P values, also below 1e-308
    XP.reset()
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        cohen_kappa_score, mean_absolute_error,
        matthews_corrcoef, balanced_accuracy_score,
        f1_score, accuracy_score, precision_score, recall_score
    )
    import json
    import warnings
    import os
    import tempfile
    import shutil
    from pathlib import Path

    warnings.filterwarnings('ignore')


    # =====================================================================
    #  FILE PATHS (set in paths.py)
    # =====================================================================
    # inputs
    PREDICTED_FILE = str(CLASSIFICATIONS)
    GT_FILE = str(GT_XLSX)
    OUTPUT_DIR = str(WORK / 'notebook')

    # one spelling per category (only letters and digits are compared)
    import re as _re
    CANON_CATEGORIES = [
        "Shortened life span: infancy", "Shortened life span: childhood/adolescence",
        "Shortened life span: adulthood", "Intellectual disability", "Internal physical malformations",
        "Impaired mobility", "Dysmorphic feature", "Sensory Impairment - Vision",
        "Sensory Impairment - Hearing", "Sensory Impairment - Touch", "Immunodeficiency/cancer",
        "Mental illness", "Infertility", "Clinical Sign or Laboratory Abnormality",
        "NFC", "Out of Scope",
    ]
    _ALIASES = {'sensoryimpairmenttouchothers': 'Sensory Impairment - Touch',
                'nfcnotfurtherclassified': 'NFC', 'nfcnotfurtherclassifiable': 'NFC',
                'outofscopeacquiredmultifactorial': 'Out of Scope'}
    def _key(t):
        return _re.sub(r'[^a-z0-9]', '', str(t).replace('â€“', '-').lower())
    _CANON = {_key(c): c for c in CANON_CATEGORIES}
    _CANON.update(_ALIASES)
    def canon_category(x):
        return np.nan if pd.isna(x) else _CANON.get(_key(x), np.nan)


    # =====================================================================
    #  SECTION 1 : SHARED STYLING
    # =====================================================================
    def setup_plot_style():
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.linewidth'] = 1
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.titlepad'] = 12
        plt.rcParams['xtick.major.width'] = 1
        plt.rcParams['ytick.major.width'] = 1
        plt.rcParams['xtick.major.size'] = 4
        plt.rcParams['ytick.major.size'] = 4
        plt.rcParams['xtick.minor.width'] = 0.5
        plt.rcParams['ytick.minor.width'] = 0.5
        plt.rcParams['xtick.minor.size'] = 2
        plt.rcParams['ytick.minor.size'] = 2
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        nature_colors = [
            '#0072B2', '#DF8F44', '#00A1D5',
            '#B24745', '#79AF97', '#6A6599', '#80796B',
        ]
        sns.set_palette(nature_colors)
        return nature_colors


    # =====================================================================
    #  SECTION 2 : STATISTICAL UTILITIES
    # =====================================================================
    def format_pvalue(log10p):
        return XP.fmt_tex(log10p)


    def calculate_chisquare(contingency_table):
        try:
            chi2, p_val, dof, _ = chi2_contingency(contingency_table)
            lp = XP.chi2(chi2, dof)
            XP.record(f'figure: chi-square {contingency_table.index.name} x {contingency_table.columns.name}',
                      'chi-square test of independence', f'chi2({dof}) = {chi2:.2f}', lp)
            return f"χ²({dof}) = {chi2:,.2f}, {format_pvalue(lp)}", chi2, p_val, dof
        except Exception as e:
            print(f"Warning: chi-square failed - {e}")
            return "", None, None, None


    def calculate_kruskal(df, group_col, value_col):
        try:
            groups = [g[value_col].dropna() for _, g in df.groupby(group_col)]
            if len(groups) < 2:
                return "", None, None, None
            dof = len(groups) - 1
            h_stat, p_val = kruskal(*groups)
            lp = XP.chi2(h_stat, dof)
            XP.record(f'figure: Kruskal-Wallis {value_col} by {group_col}', 'Kruskal-Wallis H test',
                      f'H({dof}) = {h_stat:.2f}', lp)
            return f"H({dof}) = {h_stat:,.2f}, {format_pvalue(lp)}", h_stat, p_val, dof
        except Exception as e:
            print(f"Warning: Kruskal-Wallis failed - {e}")
            return "", None, None, None


    def calculate_regression_stats(x, y):
        try:
            slope, intercept, r_value, p_value, _ = linregress(x, y)
            return f"R² = {r_value**2:.3f}, {format_pvalue(XP.r(r_value, len(x)))}", r_value, p_value
        except Exception as e:
            print(f"Warning: regression failed - {e}")
            return "", None, None


    def calculate_correlation(x, y, method='pearson'):
        try:
            if method == 'pearson':
                corr, p_val = pearsonr(x, y)
                name = 'r'
            else:
                corr, p_val = spearmanr(x, y)
                name = 'rho'
            return f"{name} = {corr:.2f}, {format_pvalue(XP.r(corr, len(x)))}", corr, p_val
        except Exception as e:
            print(f"Warning: correlation failed - {e}")
            return "", None, None


    # =====================================================================
    #  SECTION 3 : DATA LOADING & PREPROCESSING
    # =====================================================================
    def load_data(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            return df
        except FileNotFoundError:
            print(f"ERROR: file not found at {file_path}")
            return pd.DataFrame()


    def reclassify_clinical_signs_as_nfc(df):
        df['original_classification_category'] = df['classification_category'].copy()
        df['original_classification_tier'] = df['classification_tier'].copy()
        mask = df['classification_category'] == 'Clinical Sign or Laboratory Abnormality'
        df.loc[mask, 'classification_category'] = 'NFC - Clinical Signs'
        df.loc[mask, 'classification_tier'] = 'NFC'
        print(f"Reclassified {mask.sum()} rows to 'NFC - Clinical Signs'")
        return df


    def preprocess_data(df):
        print("Preprocessing data...")
        df.columns = df.columns.str.strip()

        if 'severity_assessment' in df.columns and 'QoL_assessment' not in df.columns:
            df.rename(columns={'severity_assessment': 'QoL_assessment'}, inplace=True)
        if 'QoL_assessment' in df.columns:
            df['QoL_assessment'] = df['QoL_assessment'].replace('NFC', 'Not Affected')
            df['QoL_assessment'] = df['QoL_assessment'].str.title()

        def extract_mgmt_categories(text):
            if pd.isna(text):
                return []
            try:
                data = json.loads(text)
                categories = data.get("management_category", [])
                clean_cats = set()
                for cat in categories:
                    if 'Surgical' in cat:
                        clean_cats.add('Surgical')
                    elif 'Pharmacological' in cat:
                        clean_cats.add('Pharmacological')
                    elif 'Therapeutic Support' in cat:
                        clean_cats.add('Therapeutic Support')
                    elif 'Dietary' in cat:
                        clean_cats.add('Dietary')
                    elif 'Assistive' in cat or 'Devices' in cat:
                        clean_cats.add('Assistive Devices')
                    elif 'Monitoring' in cat or 'Surveillance' in cat:
                        clean_cats.add('Monitoring')
                    elif 'Genetic' in cat:
                        clean_cats.add('Genetic Counseling')
                    elif 'Radiation' in cat:
                        clean_cats.add('Radiation Therapy')
                    elif 'Conservative' in cat:
                        clean_cats.add('Conservative Management')
                    elif 'Palliative' in cat:
                        clean_cats.add('Palliative')
                    elif 'No Treatment' in cat:
                        clean_cats.add('No Treatment')
                    elif 'Information not available' not in cat:
                        clean_cats.add(cat)
                return list(clean_cats)
            except (json.JSONDecodeError, TypeError):
                return []

        df['management_categories'] = df['management_profile'].apply(extract_mgmt_categories)
        df['num_interventions'] = df['management_categories'].apply(
            lambda c: len([x for x in c if x != 'No Treatment']))

        def parse_verification_log(log_string):
            if pd.isna(log_string):
                return {}
            try:
                log_data = json.loads(log_string)
                for item in log_data:
                    if isinstance(item, dict) and 'total_claims' in item:
                        return item
            except (json.JSONDecodeError, TypeError):
                pass
            return {}

        verification_data = df['verification_log'].apply(parse_verification_log).apply(pd.Series)
        for col in ['total_claims', 'direct_support', 'valid_inferences',
                    'weak_inferences', 'unsupported', 'weighted_score']:
            if col in verification_data.columns:
                df[f'verify_{col}'] = pd.to_numeric(
                    verification_data[col], errors='coerce').fillna(0)
            else:
                df[f'verify_{col}'] = 0

        df['direct_evidence_ratio'] = np.where(
            df['verify_total_claims'] > 0,
            df['verify_direct_support'] / df['verify_total_claims'], np.nan)
        df['unsupported_ratio'] = np.where(
            df['verify_total_claims'] > 0,
            df['verify_unsupported'] / df['verify_total_claims'], np.nan)

        qol_domain_mapping = {
            'Physical':  'Physical Functioning',
            'Cognitive': 'School/Cognitive Functioning',
            'Symptoms':  'Physical Symptoms',
            'Emotional': 'Emotional Functioning',
            'Social':    'Social Functioning',
        }

        def parse_acog_criteria(criteria_str):
            if pd.isna(criteria_str):
                return {'qol': [], 'early_onset': False}
            criteria = str(criteria_str).split(', ')
            result = {'qol': set(), 'early_onset': False}
            for c in criteria:
                if 'Early Onset' in c:
                    result['early_onset'] = True
                elif 'Physical Functioning' in c:
                    result['qol'].add(qol_domain_mapping['Physical'])
                elif 'Cognitive' in c or 'School' in c:
                    result['qol'].add(qol_domain_mapping['Cognitive'])
                elif 'Physical Symptoms' in c:
                    result['qol'].add(qol_domain_mapping['Symptoms'])
                elif 'Emotional' in c:
                    result['qol'].add(qol_domain_mapping['Emotional'])
                elif 'Social' in c:
                    result['qol'].add(qol_domain_mapping['Social'])
            return {k: list(v) if isinstance(v, set) else v for k, v in result.items()}

        acog_parsed = df['acog_criteria_met_names'].apply(parse_acog_criteria)
        df['qol_domains']     = acog_parsed.apply(lambda x: x['qol'])
        df['has_early_onset'] = acog_parsed.apply(lambda x: x['early_onset'])
        df['num_qol_domains'] = df['qol_domains'].apply(len)

        df['review_status'] = df['review_status'].str.strip()
        print("Preprocessing complete.")
        return df


    # =====================================================================
    #  SECTION 4 : ANALYSIS FIGURES (paper Figures 3 to 6, S2, S3 C and D)
    # =====================================================================

    def create_clinical_landscape_panel(df, colors, output_dir):
        fig = plt.figure(figsize=(18, 21), layout='constrained')
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.2], height_ratios=[1.2, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        category_tier = pd.crosstab(df['classification_category'].fillna('NFC (Not Further Classified)'),
                                    df['classification_tier'])
        category_tier['total'] = category_tier.sum(axis=1)
        category_tier.sort_values('total', ascending=False, inplace=True)
        category_tier_pct = (category_tier.drop('total', axis=1)
                             .div(category_tier['total'], axis=0) * 100)
        category_tier_pct.plot(kind='barh', stacked=True, ax=ax1, colormap='viridis', width=0.8)
        for i, tick in enumerate(ax1.get_yticklabels()):
            total = category_tier.loc[tick.get_text(), 'total']
            ax1.text(102, i, f'N={total}', va='center', ha='left', fontsize=8)
        ax1.set_xlim(0, max(115, ax1.get_xlim()[1]))
        style.set_category_labels(ax1)
        ax1.set_xlabel('Percentage of Phenotypes (%)')
        ax1.set_ylabel('Classification Category')
        ax1.set_title('a   Tier Distribution & Volume Within Each Category',
                      loc='left', fontsize=12)
        ax1.legend(title='Tier', frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')

        df_onset = df[~df['classification_category'].str.contains(
            'Out of Scope|NFC', na=False, case=False)]
        prevalence = df_onset.groupby('classification_category')['has_early_onset'].mean() * 100
        prevalence.sort_values(ascending=True).plot(kind='barh', ax=ax2, color=colors[4])
        style.set_category_labels(ax2)
        ax2.set_xlabel('Phenotypes with Early Onset (%)')
        ax2.set_ylabel('')
        contingency = pd.crosstab(df_onset['classification_category'], df_onset['has_early_onset'])
        chi_str, *_ = calculate_chisquare(contingency)
        ax2.set_title(f'b   Early Onset Prevalence by Category ({chi_str})',
                      loc='left', fontsize=12)

        df_cat = df[~df['classification_category'].str.contains('NFC|Out of Scope', na=False)]
        df_exp_cat = df_cat.explode('management_categories', ignore_index=True)
        df_exp_cat = df_exp_cat[df_exp_cat['management_categories'] != 'Genetic Counseling']
        if not df_exp_cat['management_categories'].dropna().empty:
            cat_mgmt = pd.crosstab(df_exp_cat['classification_category'],
                                   df_exp_cat['management_categories'])
            cat_mgmt_pct = cat_mgmt.div(cat_mgmt.sum(axis=1), axis=0) * 100
            cat_mgmt_pct.plot(kind='barh', stacked=True, ax=ax3, colormap='tab20', width=0.8)
            style.set_category_labels(ax3)
        ax3.set_title('c   Management Profile Across Categories', loc='left', fontsize=12)
        ax3.set_xlabel('Proportion of Management Types (%)')
        ax3.set_ylabel('Classification Category')
        ax3.legend(title='Management Type', frameon=False,
                   bbox_to_anchor=(1.02, 1), loc='upper left')

        df_tier = df[df['classification_tier'].astype(str) != 'NFC']
        df_exp_tier = df_tier.explode('management_categories', ignore_index=True)
        df_exp_tier = df_exp_tier[df_exp_tier['management_categories'] != 'Genetic Counseling']
        tier_mgmt = pd.crosstab(df_exp_tier['classification_tier'],
                                df_exp_tier['management_categories'])
        tier_mgmt_pct = tier_mgmt.div(tier_mgmt.sum(axis=1), axis=0) * 100
        tier_mgmt_pct.plot(kind='bar', stacked=True, ax=ax4, colormap='tab20', width=0.7)
        ax4.set_title('d   Management Profile Across Tiers', loc='left', fontsize=12)
        ax4.set_ylabel('Proportion of Management Types (%)')
        ax4.set_xlabel('Classification Tier')
        ax4.legend(title='Management Type', frameon=False,
                   bbox_to_anchor=(1.01, 1), loc='upper left')
        ax4.tick_params(axis='x', rotation=0)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', linestyle='--', alpha=0.4)

        style.save(fig, os.path.join(output_dir, 'Fig3_Clinical_Landscape_and_Onset.pdf'))

        style.save(fig, os.path.join(output_dir, 'Fig3_Clinical_Landscape_and_Onset.png'))   # PNG too
        plt.show()
        print("Figure 3 saved (paper numbering).")


    def create_evidence_quality_panel(df, colors, output_dir):
        df_f = df[~df['classification_tier'].astype(str).str.contains('NFC', na=False)
                  & ~df['classification_category'].str.contains('NFC', na=False)]

        fig = plt.figure(figsize=(16, 15), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)
        ev_cols = ['verify_direct_support', 'verify_valid_inferences',
                   'verify_weak_inferences', 'verify_unsupported']
        cmap = plt.get_cmap('cividis')

        ax1 = fig.add_subplot(gs[0, 0])
        ev_tier = df_f.groupby('classification_tier')[ev_cols].sum()
        ev_tier_pct = ev_tier.div(ev_tier.sum(axis=1), axis=0) * 100
        ev_tier_pct.plot(kind='barh', stacked=True, ax=ax1, colormap=cmap, legend=False)
        ax1.set_xlabel('Proportion of Evidence Claims (%)')
        ax1.set_ylabel('Classification Tier')
        ax1.set_title('a   Evidence Profile by Tier', loc='left', fontsize=12)
        for n, (index, row) in enumerate(ev_tier_pct.iterrows()):
            cum = 0
            for i, (col, w) in enumerate(row.items()):
                if w > 5:
                    bar_color = cmap(i / len(row))
                    lum = 0.299*bar_color[0] + 0.587*bar_color[1] + 0.114*bar_color[2]
                    tc = 'white' if lum < 0.4 else 'black'
                    ax1.text(cum + w/2, n, f'{w:.0f}%',
                             ha='center', va='center', color=tc, fontsize=8, weight='bold')
                cum += w

        ax2 = fig.add_subplot(gs[0, 1])
        ev_cat = df_f.groupby('classification_category')[ev_cols].sum()
        ev_cat_pct = ev_cat.div(ev_cat.sum(axis=1), axis=0) * 100
        ev_cat_pct.sort_values('verify_direct_support').plot(
            kind='barh', stacked=True, ax=ax2, colormap=cmap, legend=False)
        ax2.set_xlabel('Proportion of Evidence Claims (%)')
        ax2.set_ylabel('Classification Category')
        ax2.set_title('b   Evidence Profile by Category', loc='left', fontsize=12)

        handles, _ = ax1.get_legend_handles_labels()
        fig.legend(handles, ['Direct', 'Valid Infer', 'Weak Infer', 'Unsupported'],
                   bbox_to_anchor=(0.5, 1.0), loc='lower center',
                   ncol=4, frameon=False, title="Evidence Type", fontsize=9)

        ax3 = fig.add_subplot(gs[1, 0])
        sns.violinplot(x='classification_tier', y='statement_support_score',
                       data=df_f, ax=ax3, palette='viridis', inner='quartile',
                       order=sorted(df_f['classification_tier'].unique()), cut=0)
        ax3.set_xlabel('Classification Tier')
        ax3.set_ylabel('Statement Support Score')
        kw_str, *_ = calculate_kruskal(df_f, 'classification_tier', 'statement_support_score')
        ax3.set_title(f'c   Statement Support Score by Tier ({kw_str})',
                      loc='left', fontsize=12)

        ax4 = fig.add_subplot(gs[1, 1])
        sns.violinplot(x='statement_support_score', y='classification_category',
                       data=df_f, ax=ax4, palette='viridis', inner='quartile',
                       orient='h', cut=0)
        ax4.set_ylabel('Classification Category')
        ax4.set_xlabel('Statement Support Score')
        kw_str, *_ = calculate_kruskal(df_f, 'classification_category', 'statement_support_score')
        ax4.set_title(f'd   Statement Support Score by Category ({kw_str})',
                      loc='left', fontsize=12)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y' if ax in [ax3] else 'x', linestyle='--', alpha=0.4)

        style.save(fig, os.path.join(output_dir, 'Fig4_Evidence_Quality.pdf'))

        style.save(fig, os.path.join(output_dir, 'Fig4_Evidence_Quality.png'))   # PNG too
        plt.show()
        print("Figure 4 saved (paper numbering).")


    def create_tier_and_qol_overview_panel(df, colors, output_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), layout='constrained')

        df_f = df[~df['classification_tier'].astype(str).str.contains('NFC', na=False)
                  & ~df['classification_category'].str.contains('NFC', na=False)]
        qol_df = df_f.explode('qol_domains', ignore_index=True).dropna(subset=['qol_domains'])

        qol_tier = pd.crosstab(qol_df['classification_tier'], qol_df['qol_domains'])
        qol_tier_pct = qol_tier.div(qol_tier.sum(axis=1), axis=0) * 100
        qol_tier_pct.plot(kind='bar', stacked=True, ax=ax1, colormap='plasma')
        chi_str, *_ = calculate_chisquare(qol_tier)
        ax1.set_title(f'a   QoL Domain Distribution by Tier ({chi_str})',
                      loc='left', fontsize=12)
        ax1.set_xlabel('Classification Tier')
        ax1.set_ylabel('Percentage of QoL Domains (%)')
        ax1.tick_params(axis='x', rotation=0)
        ax1.legend(title='QoL Domain', frameon=False,
                   bbox_to_anchor=(1.02, 1), loc='upper left')

        qol_cat = pd.crosstab(qol_df['classification_category'], qol_df['qol_domains'])
        qol_cat_pct = qol_cat.div(qol_cat.sum(axis=1), axis=0) * 100
        qol_cat_pct.plot(kind='barh', stacked=True, ax=ax2, colormap='plasma')
        chi_str, *_ = calculate_chisquare(qol_cat)
        ax2.set_title(f'b   QoL Domain Distribution by Category ({chi_str})',
                      loc='left', fontsize=12)
        ax2.set_xlabel('Percentage of QoL Domains (%)')
        ax2.set_ylabel('Classification Category')
        ax2.get_legend().remove()

        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y' if ax == ax1 else 'x', linestyle='--', alpha=0.4)

        style.save(fig, os.path.join(output_dir, 'Fig5_Tier_and_QoL_Overview.pdf'))

        style.save(fig, os.path.join(output_dir, 'Fig5_Tier_and_QoL_Overview.png'))   # PNG too
        plt.show()
        print("Figure 5 saved (paper numbering).")


    def create_qol_analysis_panel(df, colors, output_dir):
        df_f = df[~df['classification_category'].str.contains('NFC|Out of Scope', na=False)
                  & ~df['classification_tier'].astype(str).str.contains('NFC', na=False)]

        fig = plt.figure(figsize=(16, 12), layout='constrained')
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        qol_cat = pd.crosstab(df_f['classification_category'], df_f['QoL_assessment'])
        qol_cat_pct = qol_cat.div(qol_cat.sum(axis=1), axis=0) * 100
        qol_cat_pct.sort_values('Affected', ascending=True).plot(
            kind='barh', stacked=True, ax=ax1, color=[colors[1], colors[2]])
        ax1.set_xlabel('Percentage of Phenotypes (%)')
        ax1.set_ylabel('Classification Category')
        chi_str, *_ = calculate_chisquare(qol_cat)
        ax1.set_title(f'a   QoL Status by Classification Category ({chi_str})',
                      loc='left', fontsize=12)
        ax1.legend(title='QoL Status', frameon=False,
                   bbox_to_anchor=(1.02, 1), loc='upper left')

        ax2 = fig.add_subplot(gs[0, 1])
        qol_tier = pd.crosstab(df_f['classification_tier'], df_f['QoL_assessment'])
        qol_tier_pct = qol_tier.div(qol_tier.sum(axis=1), axis=0) * 100
        qol_tier_pct.plot(kind='bar', stacked=True, ax=ax2,
                          color=[colors[1], colors[2]], rot=0)
        ax2.set_xlabel('Classification Tier')
        ax2.set_ylabel('Percentage of Phenotypes (%)')
        ax2.set_title('b   QoL Status by Classification Tier',
                      loc='left', fontsize=12)
        ax2.legend(title='QoL Status', frameon=False,
                   bbox_to_anchor=(1.02, 1), loc='upper left')

        ax3 = fig.add_subplot(gs[1, :])
        df_exp = df.explode('management_categories', ignore_index=True)
        df_exp = df_exp[df_exp['management_categories'].notna()
                        & (df_exp['management_categories'] != '')]
        df_exp = df_exp[~df_exp['management_categories'].isin(
            ['Genetic Counseling', 'Radiation Therapy', 'Conservative Management'])]
        qol_mgmt = pd.crosstab(df_exp['QoL_assessment'], df_exp['management_categories'])
        qol_mgmt = qol_mgmt.loc[:, qol_mgmt.sum(axis=0) > 0]
        qol_mgmt_pct = qol_mgmt.div(qol_mgmt.sum(axis=1), axis=0) * 100
        qol_mgmt_pct.T.sort_values('Affected').plot(
            kind='barh', ax=ax3, color=[colors[1], colors[2]])
        ax3.set_ylabel('Management Category')
        ax3.set_xlabel('Proportion of Phenotypes (%)')
        chi_str, *_ = calculate_chisquare(qol_mgmt)
        ax3.set_title(f'c   Management Profile by QoL Status ({chi_str})',
                      loc='left', fontsize=12)
        ax3.legend(title='QoL Status', frameon=False,
                   bbox_to_anchor=(1.02, 1), loc='upper left')

        for ax in fig.get_axes():
            ax.grid(axis='x', linestyle='--', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        style.save(fig, os.path.join(output_dir, 'FigS2_QoL_Impact_Analysis.pdf'))

        style.save(fig, os.path.join(output_dir, 'FigS2_QoL_Impact_Analysis.png'))   # PNG too
        plt.show()
        print("Supplementary Figure S2 saved.")


    def create_management_drivers_panel(df, colors, output_dir):
        fig = plt.figure(figsize=(16, 10), layout='constrained')
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        qol_mgmt = df.groupby('num_qol_domains')['num_interventions'].mean()
        sns.regplot(x=qol_mgmt.index, y=qol_mgmt.values, ax=ax1, seed=42,
                    color=colors[2], ci=95, scatter_kws={'s': 50, 'alpha': 0.8})
        ax1.set_xlabel('Number of QoL Domains Affected')
        ax1.set_ylabel('Average Number of Interventions')
        reg_str, *_ = calculate_regression_stats(qol_mgmt.index, qol_mgmt.values)
        ax1.set_title(f'a   QoL Impact on Management Complexity ({reg_str})',
                      loc='left', fontsize=12)

        ax2 = fig.add_subplot(gs[0, 1])
        qol_df = df.explode('qol_domains', ignore_index=True).dropna(subset=['qol_domains'])
        sns.countplot(data=qol_df, y='qol_domains', ax=ax2, color=colors[1],
                      order=qol_df['qol_domains'].value_counts().index)
        ax2.set_title('b   Frequency of PedsQL QoL Domains', loc='left', fontsize=12)
        ax2.set_xlabel('Count')
        ax2.set_ylabel('QoL Domain')

        ax3 = fig.add_subplot(gs[1, 0])
        df_tier = df[df['classification_tier'].astype(str) != 'NFC']
        # tiers plotted in order 1 to 4
        df_tier = df_tier.assign(classification_tier=df_tier['classification_tier'].astype(str))
        sns.pointplot(data=df_tier, x='classification_tier', y='num_interventions',
                      order=sorted(df_tier['classification_tier'].unique()),
                      ax=ax3, color=colors[3], errorbar='sd',
                      capsize=0.1, markers='D', linestyles='--')
        ax3.set_xlabel('Classification Tier')
        ax3.set_ylabel('Average Number of Interventions')
        kw_str, *_ = calculate_kruskal(df_tier, 'classification_tier', 'num_interventions')
        ax3.set_title(f'c   Management Complexity by Tier ({kw_str})',
                      loc='left', fontsize=12)

        ax4 = fig.add_subplot(gs[1, 1])
        df_cat = df[~df['classification_category'].str.contains('NFC|Out of Scope', na=False)]
        cat_mgmt = (df_cat.groupby('classification_category')['num_interventions']
                    .agg(['mean', 'sem']).dropna())
        cat_mgmt.sort_values('mean', ascending=True, inplace=True)
        # error bars are 95% CI (1.96 x SEM), as the legend states
        ax4.barh(cat_mgmt.index, cat_mgmt['mean'], xerr=1.96 * cat_mgmt['sem'],
                 color=colors[5], capsize=4, error_kw={'elinewidth': 1.5})
        ax4.set_xlabel('Average Number of Interventions')
        ax4.set_ylabel('Classification Category')
        kw_str, *_ = calculate_kruskal(df_cat, 'classification_category', 'num_interventions')
        ax4.set_title(f'd   Management Complexity by Category ({kw_str})',
                      loc='left', fontsize=12)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y' if ax in [ax1, ax3] else 'x', alpha=0.4, linestyle='--')

        style.save(fig, os.path.join(output_dir, 'Fig6_Management_Drivers.pdf'))

        style.save(fig, os.path.join(output_dir, 'Fig6_Management_Drivers.png'))   # PNG too
        plt.show()
        print("Figure 6 saved (paper numbering).")


    def create_distribution_and_efficiency_panel(df, colors, output_dir, fig=None):
        """
    Supplementary Figure S3, panels C and D, side by side.
    Panel C: histogram + KDE of statement support score.
    Panel D: scatter + regression + marginal histograms (cost vs complexity).

    Uses GridSpecFromSubplotSpec for the marginals so that the whole figure
    is drawn as one vector figure.
    """
        own = fig is None
        if own:
            fig = plt.figure(figsize=(18, 7))
            fig.subplots_adjust(left=0.06, right=0.97, top=0.88,
                                bottom=0.14, wspace=0.38)
            outer_gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.3])
        else:
            outer_gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3])

        # ---- Panel A ----
        ax_a = fig.add_subplot(outer_gs[0])
        sns.histplot(data=df, x='statement_support_score', kde=True,
                     ax=ax_a, color=colors[0], bins=30, line_kws={'lw': 1.8})
        mean_val = df['statement_support_score'].mean()
        ax_a.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                     label=f'Mean: {mean_val:.2f}')
        ax_a.legend(fontsize=10)
        ax_a.set_title('c   Distribution of Statement Support Score',   # S3 panel C
                       loc='left', fontsize=12, fontweight='bold', pad=8)
        ax_a.set_xlabel('Statement Support Score', fontweight='bold')
        ax_a.set_ylabel('Frequency', fontweight='bold')
        ax_a.spines[['top', 'right']].set_visible(False)
        ax_a.grid(axis='y', linestyle='--', alpha=0.4)

        # ---- Panel B: joint scatter + marginals ----
        inner_gs = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer_gs[1],
            width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05,
        )
        ax_top    = fig.add_subplot(inner_gs[0, 0])
        ax_joint  = fig.add_subplot(inner_gs[1, 0])
        ax_right  = fig.add_subplot(inner_gs[1, 1])
        ax_corner = fig.add_subplot(inner_gs[0, 1])
        ax_corner.set_visible(False)

        plot_df = df[
            df['react_steps'].notna() &
            df['total_cost_usd'].notna() &
            (df['react_steps'] > 0)
        ].copy()

        if not plot_df.empty:
            x = plot_df['react_steps'].values
            y = plot_df['total_cost_usd'].values

            ax_joint.scatter(x, y, alpha=0.35, s=30,
                             color=colors[5], edgecolors='k', linewidths=0.2)

            slope, intercept, r_value, p_value, _ = linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax_joint.plot(x_line, slope * x_line + intercept,
                          color=colors[3], lw=2)

            p_txt = format_pvalue(XP.r(r_value, len(x)))
            corr_str = f'r = {r_value:.2f}, {p_txt}'

            ax_top.hist(x, bins=25, color=colors[5], alpha=0.7, edgecolor='none')
            ax_top.set_xlim(ax_joint.get_xlim())
            ax_top.axis('off')

            ax_right.hist(y, bins=25, color=colors[5], alpha=0.7,
                          edgecolor='none', orientation='horizontal')
            ax_right.set_ylim(ax_joint.get_ylim())
            ax_right.axis('off')

            ax_joint.set_xlabel('Agent Complexity (ReAct Steps)', fontweight='bold')
            ax_joint.set_ylabel('Total Cost (USD)', fontweight='bold')
            ax_joint.spines[['top', 'right']].set_visible(False)
            ax_joint.grid(linestyle='--', alpha=0.3)

            ax_top.set_title(f'd   Efficiency: Cost vs. Complexity ({corr_str})',   # S3 panel D
                             loc='left', fontsize=12, fontweight='bold', pad=8)
        else:
            ax_joint.text(0.5, 0.5, 'No data available',
                          ha='center', va='center', transform=ax_joint.transAxes)
            ax_top.set_visible(False)
            ax_right.set_visible(False)

        if not own:
            return
        style.save(fig, os.path.join(output_dir, 'FigS3CD_Distribution_and_Efficiency.pdf'))
        style.save(fig, os.path.join(output_dir, 'FigS3CD_Distribution_and_Efficiency.png'))
        plt.show()
        print("Supplementary Figure S3 panels C and D saved.")


    # =====================================================================
    #  SECTION 5 : ANALYSIS TEXT REPORT
    # =====================================================================
    def generate_analysis_report(df, output_dir):
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            def w(text):
                f.write(text + '\n')

            w('=' * 80)
            w('ANALYSIS REPORT: FIGURES 3 TO 6, S2 AND S3')
            w('=' * 80)
            w(f'\nTotal Phenotypes Analyzed: {len(df):,}\n')

            w('\n1. OVERALL DATASET DISTRIBUTIONS')
            w('-' * 50)
            w('\nDistribution by Classification Tier:')
            w(df['classification_tier'].value_counts().to_string())
            w('\nDistribution by Classification Category:')
            w(df['classification_category'].value_counts().to_string())

            w('\n\n2. CLINICAL LANDSCAPE ANALYSIS')
            w('-' * 50)
            category_tier = pd.crosstab(df['classification_category'].fillna('NFC (Not Further Classified)'),
                                        df['classification_tier'])
            category_tier['total'] = category_tier.sum(axis=1)
            category_tier.sort_values('total', ascending=False, inplace=True)
            w('\n2A. TIER DISTRIBUTION BY CATEGORY (paper Figure 3A):')
            w(category_tier.to_string())
            pct = category_tier.drop('total', axis=1).div(category_tier['total'], axis=0) * 100
            for cat in pct.index:
                w(f'\n{cat}:')
                for tier in pct.columns:
                    if pct.loc[cat, tier] > 0:
                        w(f'  Tier {tier}: {pct.loc[cat, tier]:.1f}%')
                w(f'  Total N: {category_tier.loc[cat, "total"]}')

            w('\n\n2B. EARLY ONSET PREVALENCE BY CATEGORY (paper Figure 3B):')
            df_onset = df[~df['classification_category'].str.contains(
                'Out of Scope|NFC', na=False, case=False)]
            eo = df_onset.groupby('classification_category')['has_early_onset'].agg(
                ['sum', 'count', 'mean'])
            eo['percentage'] = eo['mean'] * 100
            eo = eo.sort_values('percentage', ascending=False)
            for cat, row in eo.iterrows():
                w(f'{cat}: {int(row["sum"])}/{int(row["count"])} ({row["percentage"]:.1f}%)')
            ct = pd.crosstab(df_onset['classification_category'], df_onset['has_early_onset'])
            chi2_val, p_val, dof, _ = chi2_contingency(ct)
            w(f'\nChi-square (Early Onset vs Category): chi2={chi2_val:.2f}, df={dof}, '
              f'p={XP.record("report: chi-square early onset x category", "chi-square test of independence", f"chi2({dof}) = {chi2_val:.2f}", XP.chi2(chi2_val, dof))}')

            w('\n\n3. EVIDENCE QUALITY & VALIDATION')
            w('-' * 50)
            df_f = df[~df['classification_tier'].astype(str).str.contains('NFC', na=False)
                      & ~df['classification_category'].str.contains('NFC', na=False)]
            ev_cols = ['verify_direct_support', 'verify_valid_inferences',
                       'verify_weak_inferences', 'verify_unsupported']
            ev_tier = df_f.groupby('classification_tier')[ev_cols].sum()
            w('\n3A. EVIDENCE PROFILE BY TIER (paper Figure 4A):')
            w(ev_tier.to_string())
            ev_tier_pct = ev_tier.div(ev_tier.sum(axis=1), axis=0) * 100
            for tier in ev_tier_pct.index:
                w(f'\nTier {tier}:')
                w(f'  Direct Support: {ev_tier_pct.loc[tier, "verify_direct_support"]:.1f}%')
                w(f'  Valid Inferences: {ev_tier_pct.loc[tier, "verify_valid_inferences"]:.1f}%')
                w(f'  Weak Inferences: {ev_tier_pct.loc[tier, "verify_weak_inferences"]:.1f}%')
                w(f'  Unsupported: {ev_tier_pct.loc[tier, "verify_unsupported"]:.1f}%')
            groups = [g['statement_support_score'].dropna()
                      for _, g in df_f.groupby('classification_tier')]
            if len(groups) > 1:
                h, p = kruskal(*groups)
                w(f'\nKruskal-Wallis (Support Score across Tiers): H({len(groups)-1})={h:.2f}, '
                  f'p={XP.record("report: Kruskal-Wallis support score by tier", "Kruskal-Wallis H test", f"H({len(groups)-1}) = {h:.2f}", XP.chi2(h, len(groups) - 1))}')

            # values read from Figure 4B-D, printed here so the text can cite them
            w('\nMedian support score by tier (paper Figure 4C):')
            w(df_f.groupby('classification_tier')['statement_support_score'].median().round(3).to_string())
            ev_cat = df_f.groupby('classification_category')[ev_cols].sum()
            ev_cat_pct = (ev_cat.div(ev_cat.sum(axis=1), axis=0) * 100).round(1)
            w('\n3B. EVIDENCE PROFILE BY CATEGORY, % of claims (paper Figure 4B):')
            w(ev_cat_pct.sort_values('verify_direct_support', ascending=False).to_string())
            groups_c = [g['statement_support_score'].dropna() for _, g in df_f.groupby('classification_category')]
            h, p_k = kruskal(*groups_c)
            w(f'Kruskal-Wallis (Support Score across Categories, Figure 4D): H({len(groups_c)-1})={h:.2f}, '
              f'p={XP.record("report: Kruskal-Wallis support score by category", "Kruskal-Wallis H test", f"H({len(groups_c)-1}) = {h:.2f}", XP.chi2(h, len(groups_c) - 1))}')
            w('\nSupport score: share scoring 1.0 = {:.1f}%, >= 0.97 = {:.1f}%, < 0.5 = {:.1f}%'.format(
                (df['statement_support_score'] == 1).mean() * 100, (df['statement_support_score'] >= 0.97).mean() * 100,
                (df['statement_support_score'] < 0.5).mean() * 100))
            w('ReAct steps: ' + ', '.join(f'{int(k)} steps {v:.1f}%' for k, v in
                                           (df['react_steps'].value_counts(normalize=True).sort_index() * 100).items()))

            w('\n\n4. QUALITY OF LIFE DOMAIN DISTRIBUTIONS')
            w('-' * 50)
            qol_df = df_f.explode('qol_domains', ignore_index=True).dropna(subset=['qol_domains'])
            qol_tier = pd.crosstab(qol_df['classification_tier'], qol_df['qol_domains'])
            w('\n4A. QOL DOMAIN COUNTS BY TIER (paper Figure 5A):')
            w(qol_tier.to_string())
            chi2_val, p_val, dof, _ = chi2_contingency(qol_tier)
            w(f'\nChi-square (QoL Domains vs Tier): chi2={chi2_val:.2f}, df={dof}, '
              f'p={XP.record("report: chi-square QoL domain x tier", "chi-square test of independence", f"chi2({dof}) = {chi2_val:.2f}", XP.chi2(chi2_val, dof))}')
            w('\nOverall QoL Domain Frequency:')
            w(qol_df['qol_domains'].value_counts().to_string())

            w('\n\n5. QUALITY OF LIFE IMPACT ANALYSIS')
            w('-' * 50)
            df_f2 = df[~df['classification_category'].str.contains('NFC|Out of Scope', na=False)
                       & ~df['classification_tier'].astype(str).str.contains('NFC', na=False)]
            qol_cat = pd.crosstab(df_f2['classification_category'], df_f2['QoL_assessment'])
            w('\n5A. QOL ASSESSMENT BY CATEGORY (paper Supplementary Figure S2A):')
            w(qol_cat.to_string())
            chi2_val, p_val, dof, _ = chi2_contingency(qol_cat)
            w(f'\nChi-square: chi2={chi2_val:.2f}, df={dof}, '
              f'p={XP.record("report: chi-square QoL status x category", "chi-square test of independence", f"chi2({dof}) = {chi2_val:.2f}", XP.chi2(chi2_val, dof))}')
            n_rad = int(df['management_categories'].apply(lambda c: 'Radiation Therapy' in c).sum())
            w(f'\n5C. Phenotypes with Radiation Therapy in the management profile (left out of Supplementary Figure S2C): {n_rad}')

            w('\n\n6. MANAGEMENT STRATEGY DRIVERS')
            w('-' * 50)
            qol_mgmt_data = df.groupby('num_qol_domains')['num_interventions'].agg(
                ['mean', 'std', 'count'])
            w('\n6A. INTERVENTIONS BY QOL DOMAIN COUNT (paper Figure 6A):')
            for d, row in qol_mgmt_data.iterrows():
                w(f'  {int(d)} domains: {row["mean"]:.3f} +/- {row["std"]:.3f} '
                  f'(n={int(row["count"])})')
            slope, intercept, r, p, _ = linregress(qol_mgmt_data.index, qol_mgmt_data['mean'])
            w(f'\nLinear Regression: R2={r**2:.3f}, slope={slope:.3f}, '
              f'p={XP.record("report: linear regression of mean interventions on QoL domain count", "t test of the slope", f"R2 = {r**2:.3f}, n = {len(qol_mgmt_data)}", XP.r(r, len(qol_mgmt_data)), len(qol_mgmt_data))}')

            # numbers quoted in the text
            w('\n6B. QOL DOMAIN COUNTS, ALL PHENOTYPES (paper Figure 6B):')
            w(df.explode('qol_domains', ignore_index=True).dropna(subset=['qol_domains'])['qol_domains']
              .value_counts().to_string())
            qa = df['QoL_assessment'].astype(str).str.strip().str.lower()
            aff, unaff = df.loc[qa == 'affected', 'num_interventions'], df.loc[qa == 'not affected', 'num_interventions']
            u, p_u = mannwhitneyu(aff, unaff, alternative='two-sided')
            w(f'\nInterventions, QoL affected vs not affected: mean {aff.mean():.2f} (n={len(aff)}) vs '
              f'{unaff.mean():.2f} (n={len(unaff)}); Mann-Whitney U={u:.0f}, '
              f'p={XP.record("report: Mann-Whitney interventions, QoL affected vs not affected", "Mann-Whitney U test", f"U = {u:.0f}", XP.mann_whitney(aff, unaff), len(aff) + len(unaff))}')
            df_cat = df_f.dropna(subset=['classification_category'])
            w('\n6D. INTERVENTIONS BY CATEGORY (paper Figure 6D):')
            w(df_cat.groupby('classification_category')['num_interventions'].agg(['mean', 'std', 'count'])
              .sort_values('mean', ascending=False).round(3).to_string())
            groups_c = [g['num_interventions'].dropna() for _, g in df_cat.groupby('classification_category')]
            h, p_k = kruskal(*groups_c)
            w(f'Kruskal-Wallis: H({len(groups_c)-1})={h:.2f}, p={XP.record("report: Kruskal-Wallis interventions by category", "Kruskal-Wallis H test", f"H({len(groups_c)-1}) = {h:.2f}", XP.chi2(h, len(groups_c) - 1))}')

            df_tier = df[df['classification_tier'].astype(str) != 'NFC']
            tier_stats = df_tier.groupby('classification_tier')['num_interventions'].describe()
            w('\n6C. INTERVENTIONS BY TIER (paper Figure 6C):')
            w(tier_stats.to_string())
            groups_t = [g['num_interventions'].dropna()
                        for _, g in df_tier.groupby('classification_tier')]
            if len(groups_t) > 1:
                h, p = kruskal(*groups_t)
                w(f'\nKruskal-Wallis: H({len(groups_t)-1})={h:.2f}, '
                  f'p={XP.record("report: Kruskal-Wallis interventions by tier", "Kruskal-Wallis H test", f"H({len(groups_t)-1}) = {h:.2f}", XP.chi2(h, len(groups_t) - 1))}')

            w('\n\n7. DISTRIBUTION AND EFFICIENCY ANALYSIS')
            w('-' * 50)
            w('\nStatement Support Score Statistics:')
            w(df['statement_support_score'].describe().to_string())
            hist_vals, bins = np.histogram(df['statement_support_score'].dropna(), bins=30)
            w('\nHistogram (30 bins):')
            for i in range(len(hist_vals)):
                w(f'  [{bins[i]:.3f}, {bins[i+1]:.3f}]: {hist_vals[i]}')

            w('\n\n8. EARLY ONSET IMPACT ANALYSIS')
            w('-' * 50)
            onset    = df[df['has_early_onset']]
            no_onset = df[~df['has_early_onset']]
            w(f'With Early Onset: {len(onset)} ({len(onset)/len(df)*100:.1f}%)')
            w(f'Without Early Onset: {len(no_onset)} ({len(no_onset)/len(df)*100:.1f}%)')

            w('\n\n9. COST AND REACT STEPS (paper Figure S3D)')
            w('-' * 50)
            pairs = [
                ('total_cost_usd',        'react_steps',           'Cost vs ReAct Steps'),
            ]
            for v1, v2, label in pairs:
                if v1 in df.columns and v2 in df.columns:
                    clean = df[[v1, v2]].dropna()
                    if len(clean) > 1:
                        r_p, p_p = pearsonr(clean[v1], clean[v2])
                        r_s, p_s = spearmanr(clean[v1], clean[v2])
                        w(f'\n{label}:')
                        w(f'  Pearson r={r_p:.3f}, p={XP.record(f"report: Pearson {label}", "Pearson correlation", f"r = {r_p:.3f}", XP.r(r_p, len(clean)), len(clean))}')
                        w(f'  Spearman rho={r_s:.3f}, p={XP.fmt(XP.r(r_s, len(clean)))}')
                        w(f'  N={len(clean)}')

            w('\n\n10. OPERATIONAL ANALYTICS')
            w('-' * 50)
            if 'total_cost_usd' in df.columns:
                w('\nTotal Cost (USD):')
                w(df['total_cost_usd'].describe(
                    percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]
                ).round(6).to_string())
            if 'react_steps' in df.columns:
                w('\nReAct Steps:')
                w(df['react_steps'].describe(
                    percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]
                ).round(2).to_string())
            ok_n = (df['review_status'] == 'OK').sum()
            w(f'\nReview Status OK: {ok_n}/{len(df)} ({ok_n/len(df)*100:.1f}%)')
            w('\nReview Status Breakdown:')
            for status, cnt in df['review_status'].value_counts().items():
                w(f'  {status}: {cnt} ({cnt/len(df)*100:.1f}%)')

            w('\n\n' + '=' * 80)
            w('LEGEND: *** p<0.001   ** p<0.01   * p<0.05   ns p>=0.05')
            w('=' * 80)
            w('END OF REPORT')
            w('=' * 80)

        print(f"Analysis report saved: {report_path}")


    # =====================================================================
    #  SECTION 6 : VALIDATION LABEL MAP & UTILITIES
    # =====================================================================
    LABEL_MAP = {
        'Clinical Sign or Laboratory Abnormality':    'Clinical Sign or Laboratory Abnormality',
        'Dysmorphic feature':                         'Dysmorphic feature',
        'Immunodeficiency/cancer':                    'Immunodeficiency/cancer',
        'Impaired mobility':                          'Impaired mobility',
        'Infertility':                                'Infertility',
        'Intellectual disability':                    'Intellectual disability',
        'Internal physical malformations':            'Internal physical malformations',
        'Mental illness':                             'Mental illness',
        'NFC (Not Further Classifiable)':             'NFC',
        'Out of Scope/Medical/Environmental':         'Out of Scope',
        'Sensory Impairment - Hearing':               'Sensory impairment: hearing',
        'Out of Scope':                               'Out of Scope',
        'NFC':                                        'NFC',
        'Sensory Impairment - Touch':                 'Sensory impairment: touch, others',
        'Sensory Impairment - Vision':                'Sensory impairment: vision',
        'Sensory impairment - touch, smell, taste':   'Sensory impairment: touch, others',
        'Shortened life span: adulthood':             'Shortened life span: adulthood',
        'Shortened life span: childhood/adolescence': 'Shortened life span: childhood/adolescence',
        'Shortened life span: infancy':               'Shortened life span: infancy',
    }


    def short_label(label):
        clean = str(label).strip()
        if clean in LABEL_MAP:
            return LABEL_MAP[clean]
        return clean


    def calculate_psi(expected, actual):
        exp_dist = expected.value_counts(normalize=True).rename('expected')
        act_dist = actual.value_counts(normalize=True).rename('actual')
        psi_df = (pd.merge(exp_dist, act_dist, left_index=True, right_index=True, how='outer')
                  .replace(0, 0.0001).fillna(0.0001))
        psi_df['psi'] = ((psi_df['actual'] - psi_df['expected'])
                         * np.log(psi_df['actual'] / psi_df['expected']))
        return psi_df['psi'].sum()


    def interpret_psi(psi):
        if psi < 0.1:
            return f'No significant shift (PSI = {psi:.4f})'
        elif psi < 0.25:
            return f'Minor shift (PSI = {psi:.4f})'
        return f'Major shift (PSI = {psi:.4f})'


    def calculate_overall_metrics(y_true_cat, y_pred_cat):
        return {
            'accuracy':           accuracy_score(y_true_cat, y_pred_cat),
            'precision_weighted': precision_score(y_true_cat, y_pred_cat,
                                                  average='weighted', zero_division=0),
            'recall_weighted':    recall_score(y_true_cat, y_pred_cat,
                                               average='weighted', zero_division=0),
            'f1_weighted':        f1_score(y_true_cat, y_pred_cat,
                                           average='weighted', zero_division=0),
            'precision_macro':    precision_score(y_true_cat, y_pred_cat,
                                                  average='macro', zero_division=0),
            'recall_macro':       recall_score(y_true_cat, y_pred_cat,
                                               average='macro', zero_division=0),
            'f1_macro':           f1_score(y_true_cat, y_pred_cat,
                                           average='macro', zero_division=0),
            'mcc':                matthews_corrcoef(y_true_cat, y_pred_cat),
        }


    def get_top_misclassifications(y_true_cat, y_pred_cat, cat_labels, n=10):
        cm = confusion_matrix(y_true_cat, y_pred_cat, labels=cat_labels)
        np.fill_diagonal(cm, 0)
        errors = []
        for i, tl in enumerate(cat_labels):
            for j, pl in enumerate(cat_labels):
                if cm[i, j] > 0:
                    errors.append({'True': tl, 'Predicted': pl, 'Count': cm[i, j]})
        top = sorted(errors, key=lambda x: x['Count'], reverse=True)[:n]
        lines = ['\n' + '='*80, 'TOP MISCLASSIFICATION PAIRS', '='*80]
        for e in top:
            lines.append(f"  True '{e['True']}' -> Predicted '{e['Predicted']}': "
                         f"{e['Count']} times")
        return '\n'.join(lines)


    def generate_validation_report(y_true_cat, y_pred_cat, y_true_tier, y_pred_tier,
                                    cat_labels, psi_value, comparison_df):
        parts = []
        parts.append('=' * 80)
        parts.append('SAMPLE REPRESENTATIVENESS ANALYSIS')
        parts.append('=' * 80)
        parts.append(interpret_psi(psi_value))
        parts.append('\nDistribution Comparison:')
        parts.append(comparison_df.to_string())
        parts.append('\n' + '-'*40 + '\n')

        parts.append('=' * 80)
        parts.append('OVERALL MODEL PERFORMANCE')
        parts.append('=' * 80)
        m = calculate_overall_metrics(y_true_cat, y_pred_cat)
        for k, v in m.items():
            parts.append(f'  {k}: {v:.4f}')
        parts.append('\n' + '-'*40 + '\n')

        parts.append('=' * 80)
        parts.append('MULTI-CLASS CATEGORICAL PERFORMANCE')
        parts.append('=' * 80)
        parts.append(f'  MCC: {matthews_corrcoef(y_true_cat, y_pred_cat):.4f}')
        parts.append(
            f'  Balanced Accuracy: {balanced_accuracy_score(y_true_cat, y_pred_cat):.4f}')
        parts.append(
            f'  Macro F1: {f1_score(y_true_cat, y_pred_cat, average="macro", zero_division=0):.4f}')
        parts.append(f"  Cohen's Kappa: {cohen_kappa_score(y_true_cat, y_pred_cat):.4f}")
        parts.append('\n' + '-'*40 + '\n')

        parts.append('DETAILED PER-CLASS REPORT\n')
        parts.append(classification_report(y_true_cat, y_pred_cat,
                                           labels=cat_labels, zero_division=0))
        parts.append(get_top_misclassifications(y_true_cat, y_pred_cat, cat_labels))
        parts.append('\n' + '-'*40 + '\n')

        parts.append('=' * 80)
        parts.append('ORDINAL SEVERITY TIER PERFORMANCE')
        parts.append('=' * 80)
        mae_s   = mean_absolute_error(y_true_tier, y_pred_tier)
        kappa_q = cohen_kappa_score(y_true_tier, y_pred_tier, weights='quadratic')
        tol_acc = np.mean(np.abs(y_true_tier - y_pred_tier) <= 1) * 100
        sp_corr, _ = spearmanr(y_true_tier, y_pred_tier)
        kt_corr, _ = kendalltau(y_true_tier, y_pred_tier)
        parts.append(f'  Quadratic-Weighted Kappa: {kappa_q:.4f}')
        parts.append(f'  MAE: {mae_s:.4f}')
        parts.append(f'  Tolerance Accuracy (+/-1): {tol_acc:.2f}%')
        parts.append(f'  Spearman: {sp_corr:.4f}')
        parts.append(f'  Kendall Tau: {kt_corr:.4f}')

        f1_mac   = f1_score(y_true_cat, y_pred_cat, average='macro', zero_division=0)
        max_diff = max(y_true_tier.max() - y_true_tier.min(), 1)
        joint    = 0.3 * f1_mac + 0.7 * (1 - mae_s / max_diff)
        parts.append(f'\n  Joint Score (0.3*F1 + 0.7*Severity): {joint:.4f}')

        tmp_df = pd.DataFrame({'ytc': y_true_cat, 'ypc': y_pred_cat, 'ytt': y_true_tier})
        tmp_df['correct'] = tmp_df['ytc'] == tmp_df['ypc']
        parts.append('\nCategory Accuracy by Tier:')
        for tier, acc in sorted(tmp_df.groupby('ytt')['correct'].mean().items()):
            parts.append(f'  Tier {tier}: {acc*100:.2f}%')

        return '\n'.join(parts)


    # =====================================================================
    #  SECTION 7 : VALIDATION DASHBOARD PANELS  (A-F)
    # =====================================================================

    def _val_style():
        plt.rcParams.update({
            'font.family':      'DejaVu Sans',
            'axes.titleweight': 'bold',
            'figure.dpi':       150,
            'axes.labelsize':   15,
            'xtick.labelsize':  13,
            'ytick.labelsize':  13,
            'legend.fontsize':  12,
        })


    PANEL_SIZE = (10, 8.5)   # Figure 2 panels, portrait layout


    def _panel_axes(ax):
        """A new figure for a single panel, or the axes of a composite figure."""
        if ax is None:
            fig, ax = plt.subplots(figsize=PANEL_SIZE)
            return fig, ax, True
        return ax.figure, ax, False


    def compose_figure2(calls, output_path):
        """Figure 2 as one vector figure, three rows of two panels at full page width."""
        _val_style()
        fig = plt.figure(figsize=(PANEL_SIZE[0] * 2, PANEL_SIZE[1] * 2.8), layout='constrained')
        for sf, (func, args) in zip(fig.subfigures(3, 2).flat, calls):
            func(*args, ax=sf.add_subplot())
        style.save(fig, output_path)
        style.save(fig, output_path.replace('.pdf', '.png'))
        plt.close(fig)


    def fig_distribution_comparison(comparison_df, psi_interpretation, path, ax=None):
        _val_style()
        plot_df = comparison_df[['Population Proportion (%)']].copy()
        plot_df['Sample Proportion (%)'] = (
            comparison_df['Current Count'] / comparison_df['Current Count'].sum() * 100)
        plot_df.index = [short_label(x) for x in plot_df.index]
        plot_df = plot_df.sort_values('Population Proportion (%)', ascending=True)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            own = True
        else:
            fig, own = ax.figure, False
        y = np.arange(len(plot_df))
        h = 0.35
        ax.barh(y - h/2, plot_df['Population Proportion (%)'], h,
                label='Population', color='#2c7bb6', edgecolor='white')
        ax.barh(y + h/2, plot_df['Sample Proportion (%)'], h,
                label='Sample (GT)', color='#abd9e9', edgecolor='white')
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df.index, fontsize=14)
        ax.set_xlabel('Proportion (%)', fontsize=16, weight='bold')
        ax.set_title('a   Distribution Comparison\n(Sample vs. Population)',   # Supplementary Figure S3 A
                     fontsize=18, weight='bold', loc='left')
        ax.legend(fontsize=12, loc='upper right',
                  bbox_to_anchor=(1.0, -0.1), ncol=2, frameon=True)
        ax.grid(axis='x', ls='--', alpha=0.4)
        if own:
            fig.subplots_adjust(bottom=0.2)
            style.save(fig, path, width_mm=88)
            plt.close(fig)


    def fig_overall_performance(y_true_cat, y_pred_cat, path, ax=None):
        _val_style()
        m = calculate_overall_metrics(y_true_cat, y_pred_cat)
        names = ['Accuracy', 'Prec.\n(Wt)', 'Rec.\n(Wt)', 'F1\n(Wt)',
                 'Prec.\n(Mac)', 'Rec.\n(Mac)', 'F1\n(Mac)', 'MCC']
        vals  = [m['accuracy'], m['precision_weighted'], m['recall_weighted'],
                 m['f1_weighted'], m['precision_macro'], m['recall_macro'],
                 m['f1_macro'], m['mcc']]
        bar_colors = ['#0072B2', '#56B4E9', '#009E73', '#E69F00',
                      '#F0E442', '#D55E00', '#CC79A7', '#000000']   # Okabe-Ito, colour-blind safe
        fig, ax, own = _panel_axes(ax)
        bars = ax.bar(names, vals, color=bar_colors, edgecolor='black', lw=1.2, width=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=13, weight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score', fontsize=16, weight='bold')
        ax.set_title('a   Overall Model Performance', fontsize=18, weight='bold', loc='left')   # Figure 2 A
        ax.grid(axis='y', ls='--', alpha=0.3)
        if own:
            fig.tight_layout()
        if own:
            style.save(fig, path, width_mm=88)
            plt.close(fig)


    def fig_confusion_matrix(y_true_cat, y_pred_cat, labels, path, ax=None):
        _val_style()
        cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        short_labels = [short_label(l) for l in labels]
        n    = len(labels)
        size = max(14, n * 1.0)
        if ax is None:
            fig, ax = plt.subplots(figsize=(size + 2, size))
            own = True
        else:
            fig, own = ax.figure, False
        sns.heatmap(cm_norm * 100, annot=True, fmt='.0f', cmap='Blues',
                    xticklabels=short_labels, yticklabels=short_labels, ax=ax,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Recall Rate (%)', 'shrink': 0.8},
                    annot_kws={'size': 13}, square=True)
        ax.set_title('b   Normalized Confusion Matrix',   # Supplementary Figure S3 B
                     fontsize=18, weight='bold', loc='left', pad=14)
        ax.set_ylabel('True Category', fontsize=16, weight='bold')
        ax.set_xlabel('Predicted Category', fontsize=16, weight='bold')
        ax.tick_params(axis='x', labelsize=16, rotation=45)
        ax.tick_params(axis='y', labelsize=16, rotation=0)
        plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
        if own:
            fig.tight_layout()
            style.save(fig, path, width_mm=88)
            plt.close(fig)


    def fig_metrics_by_category(class_report_dict, path, ax=None):
        _val_style()
        df = pd.DataFrame(class_report_dict).T
        df = df.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        df = df[['precision', 'recall', 'f1-score']].sort_values('f1-score', ascending=True)
        df.index = [short_label(x) for x in df.index]
        fig, ax, own = _panel_axes(ax)
        y = np.arange(len(df))
        h = 0.25
        ax.barh(y - h, df['precision'], h, label='Precision', color='#a6cee3', edgecolor='white')
        ax.barh(y,     df['recall'],    h, label='Recall',    color='#1f78b4', edgecolor='white')
        ax.barh(y + h, df['f1-score'],  h, label='F1-Score',  color='#08306b', edgecolor='white')
        ax.set_yticks(y)
        ax.set_yticklabels(df.index, fontsize=14)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Score', fontsize=16, weight='bold')
        ax.set_title('b   Performance Metrics by Category',   # Figure 2 B
                     fontsize=18, weight='bold', loc='left')
        ax.legend(fontsize=12, loc='upper right',
                  bbox_to_anchor=(1.0, -0.12), ncol=3, frameon=True)
        ax.grid(axis='x', ls='--', alpha=0.4)
        if own:
            fig.subplots_adjust(bottom=0.22)
        if own:
            style.save(fig, path, width_mm=88)
            plt.close(fig)


    def fig_tier_scatter(y_true_tier, y_pred_tier, path, ax=None):
        _val_style()
        fig, ax, own = _panel_axes(ax)
        jitter = 0.18
        np.random.seed(42)   # reproducible jitter
        xt = y_true_tier + np.random.uniform(-jitter, jitter, size=len(y_true_tier))
        xp = y_pred_tier + np.random.uniform(-jitter, jitter, size=len(y_pred_tier))
        ax.scatter(xt, xp, alpha=0.35, s=60, edgecolors='k', linewidths=0.3,
                   c=y_true_tier, cmap='viridis')
        lo = min(y_true_tier.min(), y_pred_tier.min()) - 0.3
        hi = max(y_true_tier.max(), y_pred_tier.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect prediction')
        ax.set_xlabel('True Severity Tier', fontsize=16, weight='bold')
        ax.set_ylabel('Predicted Severity Tier', fontsize=16, weight='bold')
        ax.set_title('c   Jitter Scatter Plot of Tiers',   # Figure 2 C
                     fontsize=18, weight='bold', loc='left')
        ax.legend(fontsize=13)
        ax.set_aspect('equal', adjustable='box')
        ticks = sorted(set(np.unique(y_true_tier)) | set(np.unique(y_pred_tier)))   # 4
        tl = ['NFC' if t == 5 else str(t) for t in ticks]
        ax.set_xticks(ticks); ax.set_xticklabels(tl); ax.set_yticks(ticks); ax.set_yticklabels(tl)
        ax.grid(True, ls='--', alpha=0.3)
        if own:
            fig.tight_layout()
        if own:
            style.save(fig, path, width_mm=88)
            plt.close(fig)


    def fig_metrics_by_tier(tier_report_dict, path, ax=None):
        _val_style()
        df = pd.DataFrame(tier_report_dict).T
        df = df.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        df = df[['precision', 'recall', 'f1-score']]
        df.index = [str(x) if str(x) == 'NFC' else 'Tier ' + str(x) for x in df.index]
        fig, ax, own = _panel_axes(ax)
        x = np.arange(len(df))
        w = 0.25
        ax.bar(x - w, df['precision'], w, label='Precision', color='#a6cee3', edgecolor='white')
        ax.bar(x,     df['recall'],    w, label='Recall',    color='#1f78b4', edgecolor='white')
        ax.bar(x + w, df['f1-score'],  w, label='F1-Score',  color='#08306b', edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, fontsize=14)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score', fontsize=16, weight='bold')
        ax.set_title('d   Performance Metrics by Tier',   # Figure 2 D
                     fontsize=18, weight='bold', loc='left')
        ax.legend(fontsize=12, loc='lower center',
                  bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=True)
        ax.grid(axis='y', ls='--', alpha=0.3)
        if own:
            fig.subplots_adjust(bottom=0.2)
        if own:
            style.save(fig, path, width_mm=88)
            plt.close(fig)


    # Figure 2 E and F, agent against the no-retrieval baseline (03_baseline, openai_gpt-4o_own_knowledge).
    # Values and bootstrap CIs are computed from that run's per_term_results.csv (898 terms) by
    # baseline_scores_cs_nontiered() below, with the same method as baseline.py.
    BASE_COL = {'main run': '#2b6cb0', 'baseline': '#a0aec0'}
    BASE_NAME = {'main run': 'Agent', 'baseline': 'Baseline (no retrieval)'}


    def baseline_scores_cs_nontiered():
        """agent and baseline scores on the same 898 terms, recomputed from baseline_per_term_results.csv
    with Clinical Sign or Laboratory Abnormality as non-tiered (tier NFC). Same method as baseline.py:
    2,000 bootstrap resamples of terms, seed 42, percentile 95% CI, exact McNemar test, tier kappa
    quadratic weighted on the order 1, 2, 3, 4, NFC. Written to baseline_scores_CS_nontiered.csv."""
        import math
        d = pd.read_csv(BASELINE_PER_TERM, dtype=str, encoding='utf-8-sig')
        cats = sorted(set(d.gt_category) | set(d.main_category) | set(d.base_category))
        tiers = ['1', '2', '3', '4', 'NFC']
        cs = 'Clinical Sign or Laboratory Abnormality'
        for w in ('gt', 'main', 'base'):
            d.loc[d[f'{w}_category'] == cs, f'{w}_tier'] = 'NFC'

        def kappa(a, b, labels, quad):
            idx = {l: i for i, l in enumerate(labels)}; n = len(labels)
            obs = np.zeros((n, n))
            np.add.at(obs, ([idx[x] for x in a], [idx[y] for y in b]), 1)
            obs /= obs.sum(); exp = np.outer(obs.sum(1), obs.sum(0))
            i, j = np.indices((n, n))
            wt = ((i - j) ** 2) / ((n - 1) ** 2) if quad else (i != j).astype(float)
            den = (wt * exp).sum()
            return 1.0 if den == 0 else 1 - (wt * obs).sum() / den

        rng = np.random.default_rng(42)
        rows = []
        for lvl, labels, quad, kn in (('category', cats, False, "Cohen's kappa"),
                                      ('tier', tiers, True, 'quadratic weighted kappa')):
            g, m, b = (d[f'{w}_{lvl}'].to_numpy() for w in ('gt', 'main', 'base'))
            n = len(g)
            st = lambda gg, pp: ((gg == pp).mean(), kappa(gg, pp, labels, quad))
            am, km = st(g, m); ab, kb = st(g, b)
            boots = []
            for _ in range(2000):
                s_ = rng.integers(0, n, n)
                a1, k1 = st(g[s_], m[s_]); a2, k2 = st(g[s_], b[s_])
                boots.append((a1, k1, a2, k2, a1 - a2, k1 - k2))
            lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
            cm, cb = m == g, b == g
            bb, cc = int(np.sum(cm & ~cb)), int(np.sum(~cm & cb))
            pv = min(1.0, 2 * sum(math.comb(bb + cc, k) for k in range(min(bb, cc) + 1)) / 2 ** (bb + cc))
            XP.record(f'figure: McNemar agent vs baseline, {lvl} accuracy', 'exact McNemar test',
                      f'b = {bb}, c = {cc}', XP.mcnemar(bb, cc), n)
            for (who, met), v, l_, h_ in zip([('main run', 'accuracy'), ('main run', kn), ('baseline', 'accuracy'),
                                              ('baseline', kn), ('main minus baseline', 'accuracy'),
                                              ('main minus baseline', kn)],
                                             [am, km, ab, kb, am - ab, km - kb], lo, hi):
                r = {'level': lvl, 'run': who, 'metric': met, 'n': n, 'value': round(v, 4),
                     'ci95_low': round(l_, 4), 'ci95_high': round(h_, 4)}
                if who == 'main minus baseline' and met == 'accuracy':
                    r.update(mcnemar_main_right_base_wrong=bb, mcnemar_main_wrong_base_right=cc, mcnemar_p_exact=pv)
                rows.append(r)
        sc = pd.DataFrame(rows)
        sc.to_csv(os.path.join(OUTPUT_DIR, 'baseline_scores_CS_nontiered.csv'), index=False, encoding='utf-8-sig')
        return sc


    def fig_baseline_overall(path, ax=None):
        _val_style()
        sc = baseline_scores_cs_nontiered()   # Clinical Sign non-tiered
        metrics = [('category', 'accuracy', 'Category\naccuracy'), ('category', "Cohen's kappa", 'Category\nkappa'),
                   ('tier', 'accuracy', 'Tier\naccuracy'), ('tier', 'quadratic weighted kappa', 'Tier weighted\nkappa')]
        n = int(sc.n.iloc[0])
        fig, ax, own = _panel_axes(ax)
        x = np.arange(len(metrics))
        for k, run in enumerate(('main run', 'baseline')):
            r = [sc[(sc.level == l) & (sc.metric == m) & (sc.run == run)].iloc[0] for l, m, _ in metrics]
            v = np.array([t.value for t in r])
            lo = v - np.array([t.ci95_low for t in r]); hi = np.array([t.ci95_high for t in r]) - v
            pos = x + (k - 0.5) * 0.38
            ax.bar(pos, v, 0.36, color=BASE_COL[run], label=BASE_NAME[run], edgecolor='white',
                   yerr=[lo, hi], capsize=5, error_kw={'lw': 1.2})
            for p_, vv, h in zip(pos, v, hi):
                ax.text(p_, vv + h + 0.015, f'{vv:.2f}', ha='center', va='bottom', fontsize=13, weight='bold')
        ax.set_xticks(x); ax.set_xticklabels([m[2] for m in metrics], fontsize=13)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Agreement with expert ground truth', fontsize=16, weight='bold')
        ax.set_title(f'e   Agent vs. No-Retrieval Baseline (n = {n})', fontsize=18, weight='bold', loc='left')
        ax.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=True)
        ax.grid(axis='y', ls='--', alpha=0.3)
        if own:
            fig.subplots_adjust(bottom=0.25)
        if own:
            style.save(fig, path, width_mm=88)
            plt.close(fig)


    def fig_baseline_by_category(path, ax=None):
        _val_style()
        d = pd.read_csv(BASELINE_PER_TERM, dtype=str, encoding='utf-8-sig')
        d['cat'] = d['gt_category'].map(canon_category)
        for c in ('main_category_correct', 'base_category_correct'):
            d[c] = d[c] == 'True'
        per = (d.groupby('cat').agg(n=('hpo_id', 'size'), agent=('main_category_correct', 'mean'),
                                    baseline=('base_category_correct', 'mean')).sort_values('n'))
        fig, ax, own = _panel_axes(ax)
        y = np.arange(len(per)); h = 0.38
        ax.barh(y + h / 2, per.agent, h, color=BASE_COL['main run'], label=BASE_NAME['main run'], edgecolor='white')
        ax.barh(y - h / 2, per.baseline, h, color=BASE_COL['baseline'], label=BASE_NAME['baseline'], edgecolor='white')
        ax.set_yticks(y)
        ax.set_yticklabels([f'{short_label(c)} ({n})' for c, n in zip(per.index, per.n)], fontsize=14)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Category Accuracy', fontsize=16, weight='bold')
        ax.set_title('f   Accuracy by Category: Agent vs. Baseline', fontsize=18, weight='bold', loc='left')
        ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.0, -0.12), ncol=2, frameon=True)
        ax.grid(axis='x', ls='--', alpha=0.4)
        if own:
            fig.subplots_adjust(bottom=0.22)
        if own:
            style.save(fig, path, width_mm=88)
            plt.close(fig)


    def compose_figure_s3(comparison_df, psi_interp, y_true_cat, y_pred_cat, cat_labels, df, colors, output_path):
        """Supplementary Figure S3 as one vector figure: panels A and B above panels C and D."""
        _val_style()
        fig = plt.figure(figsize=(20, 17), layout='constrained')
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.1, 0.75])
        left, right = top.subfigures(1, 2, width_ratios=[0.62, 1])   # room for 16 readable heatmap cells
        fig_distribution_comparison(comparison_df, psi_interp, None, ax=left.add_subplot())
        fig_confusion_matrix(y_true_cat, y_pred_cat, cat_labels, None, ax=right.add_subplot())
        setup_plot_style()
        create_distribution_and_efficiency_panel(df, colors, None, fig=bottom)
        for ax in bottom.axes:   # panels C and D at the text sizes of panels A and B
            ax.tick_params(labelsize=14)
            ax.xaxis.label.set_size(16)
            ax.yaxis.label.set_size(16)
            for t in (ax.title, ax._left_title):
                t.set_size(18)
            if ax.get_legend():
                for t in ax.get_legend().get_texts():
                    t.set_size(12)
        style.save(fig, output_path)
        style.save(fig, output_path.replace('.pdf', '.png'))
        plt.close(fig)
        _val_style()


    # =====================================================================
    #  SECTION 8 : MAIN
    # =====================================================================
    def main():
        print('\n' + '='*80)
        print('COMBINED ANALYSIS & VALIDATION PIPELINE')
        print('='*80 + '\n')

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Outputs will be saved to: {OUTPUT_DIR}\n")

        colors = setup_plot_style()

        # ==========================================
        # PART A : ANALYSIS PIPELINE  (paper Figures 3 to 6, S2, S3)
        # ==========================================
        print('\n' + '-'*60)
        print('PART A : ANALYSIS PIPELINE')
        print('-'*60 + '\n')

        df_raw = load_data(PREDICTED_FILE)
        if df_raw.empty:
            print('ERROR: could not load predicted file. Aborting.')
            return

        # one row per term, and the single term labelled 'NFC' joins 'NFC (Not Further Classified)' (E2)
        df_raw = df_raw.drop_duplicates(subset='hpo_id')
        df_raw['classification_category'] = df_raw['classification_category'].replace(
            {'NFC': 'NFC (Not Further Classified)'})
        df_raw = reclassify_clinical_signs_as_nfc(df_raw)
        df     = preprocess_data(df_raw.copy())
        # per-phenotype table behind Figures 3 to 6 and S2, read by run.py verify
        df.assign(qol_domains=df['qol_domains'].apply(';'.join),
                  management_categories=df['management_categories'].apply(';'.join))[
            ['hpo_id', 'classification_tier', 'classification_category', 'QoL_assessment', 'qol_domains',
             'num_qol_domains', 'management_categories', 'num_interventions', 'statement_support_score']
        ].to_csv(os.path.join(OUTPUT_DIR, 'phenotype_table.csv'), index=False)

        create_clinical_landscape_panel(df, colors, OUTPUT_DIR)
        create_evidence_quality_panel(df, colors, OUTPUT_DIR)
        create_tier_and_qol_overview_panel(df, colors, OUTPUT_DIR)
        create_qol_analysis_panel(df, colors, OUTPUT_DIR)
        create_management_drivers_panel(df, colors, OUTPUT_DIR)
        create_distribution_and_efficiency_panel(df, colors, OUTPUT_DIR)
        generate_analysis_report(df, OUTPUT_DIR)

        # ==========================================
        # PART B : VALIDATION PIPELINE  (Dashboard)
        # ==========================================
        print('\n' + '-'*60)
        print('PART B : VALIDATION PIPELINE')
        print('-'*60 + '\n')

        try:
            print('Loading ground-truth file...')
            gt_df   = pd.read_excel(GT_FILE)
            pred_df = pd.read_csv(PREDICTED_FILE)

            pred_df = pred_df.rename(columns={
                'classification_tier':     'classification_tier_pred',
                'classification_category': 'classification_category_pred',
            })

            def clean_categories(s):
                return (s
                        .str.replace('\u00e2\u20ac\u201c', ' - ', regex=False)
                        .str.replace('\u2013',             ' - ', regex=False)
                        .str.replace('Sensory impairment: touch, others',
                                     'Sensory Impairment - Touch', regex=False)
                        .str.replace('NFC (Not Further Classified)', 'NFC', regex=False)
                        .str.replace('NFC (Not Further Classifiable)', 'NFC', regex=False))

            # 2 and 3: unique terms, one category spelling
            gt_df['hpo_id'] = gt_df['hpo_id'].astype(str).str.strip()
            pred_df['hpo_id'] = pred_df['hpo_id'].astype(str).str.strip()
            n_gt_rows = len(gt_df)
            gt_df = gt_df.drop_duplicates(subset=['hpo_id', 'Final classification_tier',
                                                  'Final_classification_category'])
            assert gt_df['hpo_id'].is_unique, 'GT term with conflicting duplicate labels'
            pred_df = pred_df.drop_duplicates(subset='hpo_id')
            print(f'  GT: {n_gt_rows} rows, {len(gt_df)} unique terms; predictions: {len(pred_df)} terms')
            gt_df['Final_classification_category']  = gt_df['Final_classification_category'].map(canon_category)
            pred_df['classification_category_pred'] = pred_df['classification_category_pred'].map(canon_category)

            cons_df = pred_df.copy()

            print('Merging...')
            eval_df = pd.merge(gt_df, pred_df, on='hpo_id', how='inner')
            # keep NFC terms; NFC is placed after Tier 4 (5) for the tier panels
            def _tier(x):
                s_ = str(x).strip().upper().replace('TIER', '').strip()
                s_ = s_[:-2] if s_.endswith('.0') else s_
                return 5 if s_ == 'NFC' else (int(s_) if s_ in ('1', '2', '3', '4') else np.nan)
            eval_df['Final classification_tier'] = eval_df['Final classification_tier'].map(_tier)
            eval_df['classification_tier_pred']  = eval_df['classification_tier_pred'].map(_tier)
            # Clinical Sign or Laboratory Abnormality is non-tiered (Methods), so its tier is NFC (5)
            # on each side, set from that side's own category. The data files store tier 4 for these terms.
            _cs = 'Clinical Sign or Laboratory Abnormality'
            eval_df.loc[eval_df['Final_classification_category'] == _cs, 'Final classification_tier'] = 5
            eval_df.loc[eval_df['classification_category_pred'] == _cs, 'classification_tier_pred'] = 5
            print(f"  Clinical Sign set to NFC: GT {(eval_df['Final_classification_category'] == _cs).sum()}, "
                  f"agent {(eval_df['classification_category_pred'] == _cs).sum()}")

            orig = len(eval_df)
            eval_df.dropna(subset=[
                'Final_classification_category', 'classification_category_pred',
                'Final classification_tier',     'classification_tier_pred',
            ], inplace=True)
            dropped = orig - len(eval_df)
            if dropped:
                print(f'  Dropped {dropped} rows with missing values.')

            eval_df['Final classification_tier'] = eval_df['Final classification_tier'].astype(int)
            eval_df['classification_tier_pred']  = eval_df['classification_tier_pred'].astype(int)
            print(f'  {len(eval_df)} matched and clean terms.')

            y_true_cat  = eval_df['Final_classification_category']
            y_pred_cat  = eval_df['classification_category_pred']
            y_true_tier = eval_df['Final classification_tier']
            y_pred_tier = eval_df['classification_tier_pred']
            cat_labels  = sorted(y_true_cat.unique())

            print('Computing PSI...')
            gt_psi   = gt_df.rename(columns={'Final_classification_category': 'category'}) \
                       if 'Final_classification_category' in gt_df.columns else gt_df.copy()
            cons_psi = cons_df.rename(columns={'classification_category_pred': 'category'}) \
                       if 'classification_category_pred' in cons_df.columns else cons_df.copy()

            pop_dist     = cons_psi['category'].value_counts(normalize=True)
            cur_counts   = gt_psi['category'].value_counts()
            ideal_counts = (pop_dist * len(gt_psi)).round().astype(int)
            psi_value    = calculate_psi(expected=cons_psi['category'],
                                         actual=gt_psi['category'])
            psi_interp   = interpret_psi(psi_value)

            comparison_df = pd.DataFrame({
                'Population Proportion (%)': (pop_dist * 100).round(2),
                'Current Count':  cur_counts,
                'Ideal Count':    ideal_counts,
            }).fillna(0).astype({'Current Count': int, 'Ideal Count': int})
            comparison_df['Action Required (Difference)'] = (
                comparison_df['Ideal Count'] - comparison_df['Current Count'])
            comparison_df.sort_index(inplace=True)

            report_path = os.path.join(OUTPUT_DIR, 'validation_report.txt')
            print(f'Writing validation report -> {report_path}')
            report = generate_validation_report(
                y_true_cat, y_pred_cat, y_true_tier, y_pred_tier,
                cat_labels, psi_value, comparison_df)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)

            class_report_dict = classification_report(
                y_true_cat, y_pred_cat, labels=cat_labels,
                zero_division=0, output_dict=True)
            tier_labels_int  = sorted(y_true_tier.unique())
            tier_labels_str  = ['NFC' if t == 5 else str(t) for t in tier_labels_int]
            tier_report_dict = classification_report(
                y_true_tier, y_pred_tier, labels=tier_labels_int,
                target_names=tier_labels_str, zero_division=0, output_dict=True)

            print('Generating validation figures...')
            # Figure 2: panels A to D, plus the baseline panels E and F.
            # Supplementary Figure S3: panels A and B above panels C and D.
            tmp_dir = os.path.join(OUTPUT_DIR, 'panels')
            os.makedirs(tmp_dir, exist_ok=True)
            P = {k: os.path.join(tmp_dir, f'{k}.png') for k in
                 ('F2A_overall', 'F2B_by_category', 'F2C_tier_scatter',
                  'F2D_by_tier', 'F2E_baseline_overall', 'F2F_baseline_by_category')}
            print('Assembling Figure 2 and Supplementary Figure S3...')
            dashboard_path = os.path.join(OUTPUT_DIR, 'Fig2_validation_composite.pdf')
            # one vector figure, three rows of two panels at full page width
            compose_figure2([(fig_overall_performance, (y_true_cat, y_pred_cat, P['F2A_overall'])),
                             (fig_metrics_by_category, (class_report_dict, P['F2B_by_category'])),
                             (fig_tier_scatter, (y_true_tier, y_pred_tier, P['F2C_tier_scatter'])),
                             (fig_metrics_by_tier, (tier_report_dict, P['F2D_by_tier'])),
                             (fig_baseline_overall, (P['F2E_baseline_overall'],)),
                             (fig_baseline_by_category, (P['F2F_baseline_by_category'],))],
                            dashboard_path)
            compose_figure_s3(comparison_df, psi_interp, y_true_cat, y_pred_cat, cat_labels, df, colors,
                              os.path.join(OUTPUT_DIR, 'FigS3_composite.pdf'))

            m = calculate_overall_metrics(y_true_cat, y_pred_cat)
            eval_df.to_csv(os.path.join(OUTPUT_DIR, 'validation_terms_scored.csv'), index=False, encoding='utf-8-sig')
            print(f'Scored terms: {len(eval_df)}; GT terms not in the agent output: '
                  f'{len(set(gt_df.hpo_id) - set(pred_df.hpo_id))}')
            print('\n' + '='*55)
            print(f'PSI:         {psi_interp}')
            print(f'Accuracy:    {m["accuracy"]:.4f}')
            print(f'F1 (Wt):     {m["f1_weighted"]:.4f}')
            print(f'F1 (Macro):  {m["f1_macro"]:.4f}')
            print(f'MCC:         {m["mcc"]:.4f}')
            print('='*55)

        except FileNotFoundError as e:
            print(f'\nERROR: file not found. {e}')
        except KeyError as e:
            print(f'\nERROR: missing column. {e}')
        except Exception as e:
            import traceback
            print(f'\nERROR: {e}')
            traceback.print_exc()

        XP.write(os.path.join(OUTPUT_DIR, 'exact_p_values.csv'))
        print('\n' + '='*80)
        print('PIPELINE COMPLETE')
        print('='*80)
        print(f'\nAll outputs saved to: {OUTPUT_DIR}')
        print('\nAnalysis outputs:')
        print('  Fig3_Clinical_Landscape_and_Onset.pdf')
        print('  Fig4_Evidence_Quality.pdf')
        print('  Fig5_Tier_and_QoL_Overview.pdf')
        print('  FigS2_QoL_Impact_Analysis.pdf')
        print('  Fig6_Management_Drivers.pdf')
        print('  FigS3CD_Distribution_and_Efficiency.pdf')
        print('  analysis_report.txt')
        print('\nValidation outputs:')
        print('  Fig2_validation_composite.pdf')
        print('  validation_report.txt')
        print('=' * 80 + '\n')


    main()
    reset_style()


    # ####################################################################################################
    # PART 4: copy the figures to their paper names
    # ####################################################################################################
    NAMES = {}
    def _add(src_dir, stem, dst_stem, exts=('.pdf', '.png')):
        base = FIG7_DIR if src_dir == 'figure7' else WORK / src_dir
        for e in exts:
            NAMES[base / (stem + e)] = dst_stem + e
    _add('notebook', 'Fig2_validation_composite', 'main/Figure2_validation')
    _add('notebook', 'Fig3_Clinical_Landscape_and_Onset', 'main/Figure3_clinical_landscape')
    _add('notebook', 'Fig4_Evidence_Quality', 'main/Figure4_evidence_quality')
    _add('notebook', 'Fig5_Tier_and_QoL_Overview', 'main/Figure5_QoL_domains')
    _add('notebook', 'Fig6_Management_Drivers', 'main/Figure6_management_complexity')
    _add('figure7', 'Figure7', 'main/Figure7_gene_level')
    _add('S1', '3_combined_performance_heatmap', 'supplementary/FigureS1_model_selection')
    _add('notebook', 'FigS2_QoL_Impact_Analysis', 'supplementary/FigureS2_QoL_status_management')
    _add('notebook', 'FigS3_composite', 'supplementary/FigureS3_PSI_confusion_support_efficiency')
    _add('notebook', 'validation_report', 'data_and_reports/validation_report_Figure2', ('.txt',))
    _add('notebook', 'analysis_report', 'data_and_reports/analysis_report_Figures3to6_S2_S3', ('.txt',))
    _add('notebook', 'validation_terms_scored', 'data_and_reports/validation_terms_scored', ('.csv',))
    _add('figure7', 'figure7_data', 'data_and_reports/figure7_data', ('.xlsx',))
    _add('notebook', 'baseline_scores_CS_nontiered', 'data_and_reports/baseline_scores_Figure2E', ('.csv',))
    _add('notebook', 'phenotype_table', 'data_and_reports/phenotype_table_Figures3to6_S2', ('.csv',))
    _add('notebook', 'exact_p_values', 'data_and_reports/exact_p_values', ('.csv',))
    missing = []
    for src, dst in NAMES.items():
        (OUTPUT / dst).parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, OUTPUT / dst)
        else:
            missing.append(src.name)
    print('\nFigures written to', OUTPUT)
    for src, dst in NAMES.items():
        if src.exists():
            print('  ', dst)
    if missing:
        print('MISSING (check the messages above):', ', '.join(missing))
    return OUTPUT
