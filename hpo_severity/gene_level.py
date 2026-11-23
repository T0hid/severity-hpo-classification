"""
Gene-level severity: Figure 7 (panels A to E), Table 2 and the comparison with reference gene lists.

Run through the entry point (python run.py figures or python run.py verify), after the gene table.
Reads the gene-disease-phenotype table, the MONDO, HPO, GenCC and HGNC files in data/inputs/, the
Human Intolerome gene table and the gene lists in data/gene_lists/:
  SF_ACMG_v3.2_genes_MIM.txt         ACMG SF v3.2 genes and MIM disorder numbers (Miller et al. 2023)
  CS_Gregg2021_tier3_genes_MIM.txt   ACMG Tier 3 carrier screening genes and MIM numbers (Gregg et al. 2021)
  MackenziesMission_genes.csv        Mackenzie's Mission genes with OMIM and MONDO identifiers
Writes Figure7.pdf, Figure7.png and figure7_data.xlsx to the output folder.

Rules
  Pairs        all gene-disease pairs of the preprocessing output except the 359 whose strongest GenCC
               classification is Limited (flag_limited_only). Severity: the frequency-filtered version (severity_freq).
  7A           severity of every labelled pair.
  7B, Table 2  a pair is counted under every HPO body system (the 23 children of HP:0000118 Phenotypic abnormality)
               that contains at least one of its phenotypes kept by the frequency filter.
  7C           mode of inheritance of the pair: AR, AD, X-linked, complex (Multiple or Other); pairs with no
               inheritance annotation are left out and counted.
  7D           gene level. A list gene's severity is the highest severity among the conditions the list gives for
               that gene, matched to pairs in the dataset (exact MONDO term, its replacement if obsolete, or a broader
               or narrower MONDO term of the same condition). The base rate is every gene in the analysis, with the
               highest severity over all its pairs.
  7E           gene level: share of genes whose (listed) condition has a phenotype tiered as shortened life span in
               infancy or in childhood/adolescence, among the phenotypes kept by the frequency filter.
  Intolerome   genes whose lethality acts through the fetus (Gene Effect Fetal or Maternal,Fetal) with Known or
               Emerging human evidence. Maternal-only and Unknown-effect genes are left out.
"""
import re, json, collections, pathlib, hashlib, logging
import pandas as pd
import matplotlib
import matplotlib.ticker


def run(output_dir, gene_table=None):
    """Draws Figure 7 and writes figure7_data.xlsx to output_dir. Returns the output folder."""
    from hpo_severity import style; style.reset()   # the matplotlib settings of the paper figures
    import matplotlib.pyplot as _plt; _plt.close("all")
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from . import paths

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)   # Arial falls back to a metric-compatible font if absent
    ROOT = paths.ROOT
    OUT = pathlib.Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    FILES = {
        'master': pathlib.Path(gene_table or paths.GENE_TABLE),
        'mondo': paths.MONDO,
        'hpo': paths.HPO_2026,
        'gencc': paths.GENCC,
        'hgnc': paths.HGNC,
        'sf': paths.GENE_LISTS / 'SF_ACMG_v3.2_genes_MIM.txt',
        'cs': paths.GENE_LISTS / 'CS_Gregg2021_tier3_genes_MIM.txt',
        'mm': paths.GENE_LISTS / 'MackenziesMission_genes.csv',
        'intolerome': paths.INTOLEROME,
    }
    CLASSES = ['Profound', 'Severe', 'Moderate', 'Mild']
    RANK = {c: i for i, c in enumerate(reversed(CLASSES))}          # Mild 0 ... Profound 3
    EARLY_DEATH = ('Shortened life span: infancy', 'Shortened life span: childhood')
    FOUND = ['pair_in_dataset', 'pair_in_dataset_via_update', 'related_term_in_dataset']
    LOG = []


    def note(msg=''):
        print(msg, flush=True); LOG.append(str(msg))


    def sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        return h.hexdigest()


    for k, p in FILES.items():
        note(f'input {k:11s} {p.name:42s} sha256 {sha(p)}')

    # ------------------------------------------------------------------ dataset
    M = pd.read_csv(FILES['master'], dtype=str, keep_default_na=False, encoding='utf-8-sig')
    PAIRS_ALL = M.drop_duplicates(['HGNC', 'MONDO_ID']).set_index(['HGNC', 'MONDO_ID'])
    pair_set = set(PAIRS_ALL.index)
    by_gene_all = collections.defaultdict(set)
    for h_, m_ in pair_set:
        by_gene_all[h_].add(m_)
    ROWS = M[M.HPO_ID != '']


    def analysis(version='freq', include_limited=False):
        """Pair table and phenotype rows used for one analysis version."""
        P = PAIRS_ALL if include_limited else PAIRS_ALL[PAIRS_ALL.flag_limited_only != 'True']
        keep = (ROWS.freq_kept == 'True') if version == 'freq' else (ROWS.recorded_absent == 'False')
        R = ROWS[keep]
        R = R[pd.MultiIndex.from_frame(R[['HGNC', 'MONDO_ID']]).isin(P.index)]
        early = set(map(tuple, R[R.agent_category.str.startswith(EARLY_DEATH)][['HGNC', 'MONDO_ID']].values))
        return P, R, early


    # ------------------------------------------------------------------ MONDO and HPO
    def mid_(iri):
        return iri.rsplit('/', 1)[-1].replace('MONDO_', 'MONDO:')


    g = json.loads(FILES['mondo'].read_text(encoding='utf-8'))['graphs'][0]
    label, active, replaced_by, consider = {}, set(), {}, collections.defaultdict(set)
    omim2mondo, mondo_parents = collections.defaultdict(set), collections.defaultdict(set)
    for n in g['nodes']:
        if '/MONDO_' not in n['id']:
            continue
        m = mid_(n['id'])
        label[m] = n.get('lbl', '')
        meta = n.get('meta', {})
        if meta.get('deprecated'):
            for b in meta.get('basicPropertyValues', []):
                if b['pred'].endswith('IAO_0100001'):
                    replaced_by[m] = mid_(b['val'])
                elif b['pred'].endswith('#consider') and 'MONDO' in b['val']:
                    consider[m].add(mid_(b['val']))
            continue
        active.add(m)
        for b in meta.get('basicPropertyValues', []):
            if b['pred'] == 'http://www.w3.org/2004/02/skos/core#exactMatch':
                x = re.search(r'omim\.org/entry/(\d+)$', b['val']) or re.fullmatch(r'OMIM:(\d+)', b['val'])
                if x:
                    omim2mondo[x.group(1)].add(m)
    for e in g['edges']:
        if e['pred'] == 'is_a' and '/MONDO_' in e['sub'] and '/MONDO_' in e['obj']:
            mondo_parents[mid_(e['sub'])].add(mid_(e['obj']))
    del g


    def closure(t, links, memo):
        if t not in memo:
            seen, stack = set(), list(links.get(t, ()))
            while stack:
                x = stack.pop()
                if x not in seen:
                    seen.add(x); stack.extend(links.get(x, ()))
            memo[t] = seen
        return memo[t]


    _manc = {}
    mondo_anc = lambda t: closure(t, mondo_parents, _manc)


    def omim_to_mondo(mim):
        s = omim2mondo.get(str(mim).replace('OMIM:', '').strip(), set())
        return next(iter(s)) if len(s) == 1 else None


    hp_name, hp_parents, cur = {}, collections.defaultdict(set), None
    for line in FILES['hpo'].read_text(encoding='utf-8').splitlines():
        if line.startswith('['):
            cur = '' if line == '[Term]' else None
        elif cur is not None and line.startswith('id: '):
            cur = line[4:].strip()
        elif cur and line.startswith('name: '):
            hp_name[cur] = line[6:]
        elif cur and line.startswith('is_a: '):
            hp_parents[cur].add(line[6:].split('!')[0].strip())
    SYSTEMS = sorted(t for t, ps in hp_parents.items() if 'HP:0000118' in ps)
    _hanc = {}
    hp_anc = lambda t: closure(t, hp_parents, _hanc)
    system_of = lambda t: {s for s in SYSTEMS if s == t or s in hp_anc(t)}
    note(f'HPO body systems (children of HP:0000118): {len(SYSTEMS)}')

    # ------------------------------------------------------------------ GenCC (reason column) and HGNC
    GRANK = {'Definitive': 1, 'Strong': 2, 'Moderate': 3, 'Supportive': 4, 'Limited': 5,
             'Disputed Evidence': 6, 'Refuted Evidence': 7, 'No Known Disease Relationship': 8}
    gc = pd.read_csv(FILES['gencc'], dtype=str, usecols=['gene_curie', 'disease_curie', 'classification_title'])
    gc['rank'] = gc.classification_title.map(GRANK)
    gbest = gc.groupby(['gene_curie', 'disease_curie'])['rank'].min().map({v: k for k, v in GRANK.items()}).to_dict()
    dsm = set(map(tuple, gc[gc['rank'] <= 3][['gene_curie', 'disease_curie']].values))
    mondo_pairs = set(map(tuple, PAIRS_ALL[PAIRS_ALL.pair_source == 'MONDO_germline'].index))
    genes_per_dis = collections.Counter(m for h_, m in (dsm | mondo_pairs) if m in active)

    hg = pd.read_csv(FILES['hgnc'], sep='\t', dtype=str, usecols=['hgnc_id', 'symbol', 'prev_symbol', 'alias_symbol', 'omim_id', 'entrez_id', 'status'])
    hg = hg[hg.status == 'Approved']
    SYM = {s.upper(): i for s, i in zip(hg.symbol, hg.hgnc_id)}
    ENTREZ = {e: i for e, i in zip(hg.entrez_id, hg.hgnc_id) if isinstance(e, str)}
    GENE_MIM = collections.defaultdict(set)
    for i, v in zip(hg.hgnc_id, hg.omim_id.fillna('')):
        for x in v.split('|'):
            if x: GENE_MIM[i].add(x)


    def multi(col):
        d = collections.defaultdict(set)
        for v, i in zip(hg[col].fillna(''), hg.hgnc_id):
            for s in v.split('|'):
                if s: d[s.upper()].add(i)
        return d


    PREV, ALIAS = multi('prev_symbol'), multi('alias_symbol')


    def resolve(symbol, entrez=None):
        if entrez and entrez in ENTREZ: return ENTREZ[entrez]
        s = str(symbol).strip().upper()
        if s in SYM: return SYM[s]
        if len(PREV.get(s, ())) == 1: return next(iter(PREV[s]))
        if len(ALIAS.get(s, ())) == 1: return next(iter(ALIAS[s]))
        return None


    # ------------------------------------------------------------------ matching one listed gene-condition entry
    def why_not(hid, m):
        best = gbest.get((hid, m))
        if (hid, m) in dsm and genes_per_dis[m] > 10:
            return f'GenCC {best}; removed in preprocessing as umbrella term ({genes_per_dis[m]} genes)'
        if best in ('Disputed Evidence', 'Refuted Evidence', 'No Known Disease Relationship'):
            return f'GenCC strongest classification {best}; removed in preprocessing'
        if best:
            return f'not in MONDO germline relationships; GenCC strongest classification {best}'
        return 'not in MONDO germline relationships and not in GenCC'


    def match(hid, candidates, given=True):
        """Return status, matched MONDO, reason for a gene and the MONDO term(s) of its listed condition."""
        candidates = [c for c in dict.fromkeys(candidates) if c]
        if not given:
            return 'no_condition_given', '', 'the list gives no condition identifier'
        if not hid:
            return 'gene_not_in_dataset', '', 'symbol not found in HGNC'
        if not candidates:
            return 'OMIM_not_in_MONDO', '', 'no active MONDO term has this OMIM number as exactMatch'
        for m in candidates:
            if (hid, m) in pair_set:
                return 'pair_in_dataset', m, ''
        for m in candidates:
            if m not in active and (hid, replaced_by.get(m)) in pair_set:
                return 'pair_in_dataset_via_update', replaced_by[m], f'{m} is obsolete; replaced by {replaced_by[m]}'
        for m in candidates:
            cands = {m} if m in active else ({replaced_by[m]} if m in replaced_by else set()) | consider.get(m, set())
            for d in sorted(by_gene_all.get(hid, ())):
                for c in cands:
                    if c in mondo_anc(d) or d in mondo_anc(c):
                        rel = 'narrower' if c in mondo_anc(d) else 'broader'
                        return 'related_term_in_dataset', d, (f'{m} is obsolete; ' if m not in active else '') + \
                            f'gene is paired with the {rel} term {d} ({label.get(d, "")})'
        m = candidates[0]
        reason = why_not(hid, m) if m in active else f'{m} is not active in MONDO 2026-09-01 and has no related term paired'
        return ('gene_not_in_dataset' if hid not in by_gene_all else 'pair_not_in_dataset'), '', reason


    def entry(base, hid, mim, candidates, given=True):
        st, mm, why = match(hid, candidates, given)
        first = next((c for c in candidates if c), '')
        return {**base, 'HGNC': hid or '', 'condition_MIM': mim, 'listed_MONDO': first,
                'listed_MONDO_label': label.get(first, ''), 'status': st, 'matched_MONDO': mm,
                'dataset_Disease': PAIRS_ALL.at[(hid, mm), 'Disease'] if mm else '',
                'limited_only_excluded': (PAIRS_ALL.at[(hid, mm), 'flag_limited_only'] == 'True') if mm else '',
                'severity_freq': PAIRS_ALL.at[(hid, mm), 'severity_freq'] if mm else '', 'reason': why}


    LISTS = {}
    # ACMG SF v3.2
    rows = []
    for line in FILES['sf'].read_text(encoding='utf-8').splitlines():
        if line.startswith('#') or not line.strip(): continue
        gene, gmim, dis = line.split('|')
        hid = resolve(gene)
        for mim in dis.split(','):
            rows.append(entry({'gene': gene}, hid, mim, [omim_to_mondo(mim)]))
    LISTS['ACMG SF v3.2'] = pd.DataFrame(rows)
    # ACMG Tier 3 carrier screening
    CS_MIM_NAMED = {('BCKDHB', '245600'): '248600', ('MCPH1', '651200'): '251200'}
    rows = []
    for line in FILES['cs'].read_text(encoding='utf-8').splitlines():
        if line.startswith('#') or not line.strip(): continue
        t, gene, mims = line.split('|')
        hid = resolve(gene)
        base = {'gene': gene, 'group': 'X-linked' if t == '6' else 'autosomal recessive', 'MIM_numbers_in_row': mims}
        # In two rows of Gregg et al. 2021 the MIM number does not match the condition named beside it; the
        # number of the named condition is used.
        # Table 2, BCKDHB, "Maple syrup urine disease": 248600 (the row gives 245600, Larsen-like syndrome).
        # Table 3, MCPH1, "Primary microcephaly 1, recessive": 251200 (the row gives 651200, not a disease number).
        mims = ','.join(CS_MIM_NAMED.get((gene, x), x) for x in mims.split(','))
        conds = [x for x in dict.fromkeys(mims.split(',')) if x not in GENE_MIM.get(hid, set())]
        if not conds:
            rows.append(entry(base, hid, '', [], given=False))
        for mim in conds:
            rows.append(entry(base, hid, mim, [omim_to_mondo(mim)]))
    LISTS['ACMG carrier screening'] = pd.DataFrame(rows)
    # Mackenzie's Mission (one condition per gene after removing duplicate rows)
    mm = pd.read_csv(FILES['mm'], dtype=str).fillna('')
    mm_rows = len(mm)
    mm = mm.drop_duplicates().reset_index(drop=True)
    assert mm.gene.is_unique
    rows = []
    for r in mm.itertuples(index=False):
        cands = [r.Mondo.strip() if r.Mondo.strip().startswith('MONDO:') else None,
                 omim_to_mondo(r.OMIM_ID) if r.OMIM_ID.strip() else None]
        rows.append(entry({'gene': r.gene, 'condition_in_list': r._2}, resolve(r.gene), r.OMIM_ID, cands,
                          given=bool(r.Mondo.strip() or r.OMIM_ID.strip())))
    LISTS["Mackenzie's Mission"] = pd.DataFrame(rows)
    # Human Intolerome
    it = pd.read_csv(FILES['intolerome'], sep='\t', dtype=str, keep_default_na=False)
    it_rows = len(it)
    gl = it['Gene/Locus'].str.extract(r'name:\s*([^;]+);\s*id:\s*(\d+)')
    assert gl.notna().all().all(), 'unexpected Gene/Locus format'
    it['gene'], it['entrez'] = gl[0].str.strip(), gl[1]
    it['mims'] = [list(dict.fromkeys(re.findall(r'\d{6}', p) + re.findall(r'(\d{6})\s*\(\d\)', d)))
                  for p, d in zip(it['OMIM Phenotype'], it['Disease'])]
    x_ = lambda c: it[c].str.strip().str.lower() == 'x'
    it['prenatal_lethality'] = x_('1st trimester') | x_('2nd trimester') | x_('3rd trimester')
    it['postnatal_lethality'] = x_('Postnatal lethality')
    it['fetal_effect'] = it['Gene Effect'].isin(['Fetal', 'Maternal,Fetal'])
    EVID = {'Known': 0, 'Emerging': 1, 'Candidate': 2}
    grp = it.groupby('gene', sort=False)
    itg = grp.agg(entrez=('entrez', 'first'), gene_effect=('Gene Effect', lambda s: ','.join(sorted(set(s)))),
                  fetal_effect=('fetal_effect', 'any'),
                  human_evidence=('Human Evidence', lambda s: min(s, key=EVID.get)),
                  inheritance=('Inheritance', lambda s: ','.join(sorted(set(x for x in s if x)))),
                  prenatal_lethality=('prenatal_lethality', 'any'), postnatal_lethality=('postnatal_lethality', 'any'),
                  mims=('mims', lambda s: list(dict.fromkeys(m for l in s for m in l))), rows=('entrez', 'size')).reset_index()
    itg['in_main_set'] = itg.fetal_effect & itg.human_evidence.isin(['Known', 'Emerging'])
    rows = []
    for r in itg.itertuples(index=False):
        base = {'gene': r.gene, 'gene_effect': r.gene_effect, 'human_evidence': r.human_evidence, 'inheritance': r.inheritance,
                'prenatal_lethality': r.prenatal_lethality, 'postnatal_lethality': r.postnatal_lethality,
                'in_main_set': r.in_main_set}
        hid = resolve(r.gene, r.entrez)
        if not r.mims:
            rows.append(entry(base, hid, '', [], given=False))
        for mim in r.mims:
            rows.append(entry(base, hid, mim, [omim_to_mondo(mim)]))
    INTOL_ALL = pd.DataFrame(rows)
    LISTS['Human Intolerome (fetal effect)'] = INTOL_ALL[INTOL_ALL.in_main_set].reset_index(drop=True)
    note(f"Intolerome: {it_rows} rows, {len(itg)} genes; fetal effect with Known or Emerging evidence: {int(itg.in_main_set.sum())} genes")


    # ------------------------------------------------------------------ gene-level severity
    def highest(values):
        v = [RANK[x] for x in values if x in RANK]
        return {i: c for c, i in RANK.items()}[max(v)] if v else ''


    def gene_level(P, early, version):
        sev = P['severity_' + version]
        base = collections.defaultdict(list)
        for (h_, m_), s in sev.items():
            if s:
                base[h_].append((s, (h_, m_) in early))
        out = {'All genes (base rate)': pd.DataFrame([{'gene': h_, 'severity': highest([a for a, _ in v]),
                                                       'early_death': any(b for _, b in v)} for h_, v in base.items()])}
        for name, df in LISTS.items():
            recs = []
            for gene, d in df.groupby('gene', sort=False):
                keys = [(h_, m_) for h_, m_, st in zip(d.HGNC, d.matched_MONDO, d.status) if st in FOUND and (h_, m_) in P.index]
                s = [P.at[k, 'severity_' + version] for k in keys]
                recs.append({'gene': gene, 'severity': highest(s), 'early_death': any(k in early and P.at[k, 'severity_' + version]
                                                                                       for k in keys)})
            out[name] = pd.DataFrame(recs)
        return out


    def summarise_sets(gl):
        rows = []
        for name, d in gl.items():
            lab = d[d.severity != '']
            c = lab.severity.value_counts()
            rows.append({'gene_set': name, 'genes_in_list': len(d), 'genes_labelled': len(lab),
                         **{k: int(c.get(k, 0)) for k in CLASSES},
                         **{f'{k}_pct': round(100 * c.get(k, 0) / len(lab), 1) for k in CLASSES},
                         'Severe_or_Profound_pct': round(100 * (c.get('Profound', 0) + c.get('Severe', 0)) / len(lab), 1),
                         'early_death_genes': int(lab.early_death.sum()),
                         'early_death_pct': round(100 * lab.early_death.mean(), 1)})
        return pd.DataFrame(rows)


    # ------------------------------------------------------------------ main analysis
    P, R, EARLY = analysis('freq', include_limited=False)
    L = P[P.severity_freq != '']
    note(f'Pairs in the analysis: {len(P):,} of {len(PAIRS_ALL):,} (Limited-only pairs excluded: {len(PAIRS_ALL) - len(P)}); '
         f'labelled {len(L):,} ({L.index.get_level_values("HGNC").nunique():,} genes, '
         f'{L.index.get_level_values("MONDO_ID").nunique():,} diseases); '
         f'not labelled {P[P.severity_freq == ""].status_freq.value_counts().to_dict()}')

    # 7A
    A = L.severity_freq.value_counts().reindex(CLASSES).rename('pairs').to_frame()
    A['pct'] = (100 * A.pairs / A.pairs.sum()).round(1)
    note('7A ' + ', '.join(f'{k} {v:,} ({p}%)' for k, v, p in zip(A.index, A.pairs, A.pct)))

    # 7B and Table 2
    lab_idx = set(L.index)
    Rl = R[[k in lab_idx for k in zip(R.HGNC, R.MONDO_ID)]]
    term_sys = {t: system_of(t) for t in Rl.HPO_ID.unique()}
    pair_sys = collections.defaultdict(set)
    for h_, m_, t in zip(Rl.HGNC, Rl.MONDO_ID, Rl.HPO_ID):
        pair_sys[(h_, m_)] |= term_sys[t]
    B = pd.DataFrame([(s, L.at[k, 'severity_freq']) for k, ss in pair_sys.items() for s in ss], columns=['system', 'severity'])
    B = pd.crosstab(B.system, B.severity).reindex(columns=CLASSES, fill_value=0)
    B.index = [hp_name[s] for s in B.index]
    B.insert(0, 'pairs', B[CLASSES].sum(axis=1))
    for c in CLASSES:
        B[c + '_pct'] = (100 * B[c] / B.pairs).round(1)
    B = B.sort_values(['Profound_pct', 'pairs'], ascending=[False, False])
    B.index.name = 'body_system'
    note(f'7B: {len(B)} body systems; pairs with at least one system: {len(pair_sys):,}')

    # 7C
    moi = L.MOI_group.map({'AR': 'Autosomal recessive', 'AD': 'Autosomal dominant', 'XL': 'X-linked',
                           'Multiple': 'Complex', 'Other': 'Complex'})
    C = pd.crosstab(moi.dropna(), L.severity_freq[moi.notna()]).reindex(
        index=['Autosomal dominant', 'Autosomal recessive', 'X-linked', 'Complex'], columns=CLASSES, fill_value=0)
    C.insert(0, 'pairs', C[CLASSES].sum(axis=1))
    for c in CLASSES:
        C[c + '_pct'] = (100 * C[c] / C.pairs).round(1)
    C['Severe_or_Profound_pct'] = (100 * (C.Profound + C.Severe) / C.pairs).round(1)
    no_moi = int(moi.isna().sum())
    note(f'7C: pairs without inheritance annotation (left out): {no_moi}')

    # 7D and 7E
    GL = gene_level(P, EARLY, 'freq')
    D = summarise_sets(GL)
    note('7D/7E:\n' + D[['gene_set', 'genes_in_list', 'genes_labelled', 'Profound_pct', 'Severe_or_Profound_pct', 'early_death_pct']].to_string(index=False))

    # ------------------------------------------------------------------ sensitivity
    sens = []
    for desc, ver, lim in [('main: frequency-filtered, Limited-only pairs excluded', 'freq', False),
                           ('Limited-only pairs included', 'freq', True),
                           ('no frequency filter (absent phenotypes still removed)', 'all', False)]:
        P2, R2, E2 = analysis(ver, lim)
        L2 = P2[P2['severity_' + ver] != '']
        c = L2['severity_' + ver].value_counts()
        sens.append({'analysis': desc, 'set': 'all labelled pairs (7A)', 'n': len(L2),
                     **{k + '_pct': round(100 * c.get(k, 0) / len(L2), 1) for k in CLASSES},
                     'Severe_or_Profound_pct': round(100 * (c.get('Profound', 0) + c.get('Severe', 0)) / len(L2), 1), 'early_death_pct': ''})
        for _, r in summarise_sets(gene_level(P2, E2, ver)).iterrows():
            sens.append({'analysis': desc, 'set': r.gene_set + ' (gene level)', 'n': r.genes_labelled,
                         **{k + '_pct': r[k + '_pct'] for k in CLASSES}, 'Severe_or_Profound_pct': r.Severe_or_Profound_pct,
                         'early_death_pct': r.early_death_pct})
    # list variants
    saved = dict(LISTS)
    for desc, key, df in [('Intolerome, Known evidence only', 'Human Intolerome (fetal effect)', INTOL_ALL[INTOL_ALL.in_main_set & (INTOL_ALL.human_evidence == 'Known')]),
                          ('Intolerome, maternal effect only (not comparable)', 'Human Intolerome (fetal effect)', INTOL_ALL[INTOL_ALL.gene_effect == 'Maternal']),
                          ('ACMG carrier screening, 97 autosomal recessive genes', 'ACMG carrier screening', LISTS['ACMG carrier screening'].query("group == 'autosomal recessive'"))]:
        LISTS = {key: df}
        r = summarise_sets(gene_level(P, EARLY, 'freq')).iloc[1]
        sens.append({'analysis': 'main', 'set': desc, 'n': r.genes_labelled, **{k + '_pct': r[k + '_pct'] for k in CLASSES},
                     'Severe_or_Profound_pct': r.Severe_or_Profound_pct, 'early_death_pct': r.early_death_pct})
    LISTS = saved
    # Intolerome by lethality timing and inheritance (text, not a panel)
    for desc, sel in [('Intolerome, lethal before birth only', lambda d: d.prenatal_lethality & ~d.postnatal_lethality),
                      ('Intolerome, lethal before and after birth', lambda d: d.prenatal_lethality & d.postnatal_lethality),
                      ('Intolerome, lethal after birth only', lambda d: ~d.prenatal_lethality & d.postnatal_lethality)]:
        df = LISTS['Human Intolerome (fetal effect)']
        LISTS_ = dict(LISTS); LISTS = {'x': df[sel(df)]}
        r = summarise_sets(gene_level(P, EARLY, 'freq')).iloc[1]
        LISTS = LISTS_
        sens.append({'analysis': 'main', 'set': desc, 'n': r.genes_labelled, **{k + '_pct': r[k + '_pct'] for k in CLASSES},
                     'Severe_or_Profound_pct': r.Severe_or_Profound_pct, 'early_death_pct': r.early_death_pct})
    S = pd.DataFrame(sens)

    # ------------------------------------------------------------------ list matching summary and gene table
    ms = []
    for name, df in LISTS.items():
        found = df.status.isin(FOUND).groupby(df.gene).any()
        ms.append({'list': name, 'genes': df.gene.nunique(), 'gene_condition_entries': len(df),
                   **df.status.value_counts().reindex(FOUND + ['pair_not_in_dataset', 'gene_not_in_dataset',
                                                              'OMIM_not_in_MONDO', 'no_condition_given'], fill_value=0).to_dict(),
                   'genes_with_a_listed_condition_found': int(found.sum()),
                   'genes_not_found': ', '.join(sorted(found[~found].index))})
    MS = pd.DataFrame(ms)
    G = pd.concat([d.assign(gene_set=n) for n, d in GL.items() if n != 'All genes (base rate)'], ignore_index=True)
    G = G.rename(columns={'severity': 'highest_listed_severity_freq', 'early_death': 'early_death_phenotype'})[
        ['gene_set', 'gene', 'highest_listed_severity_freq', 'early_death_phenotype']]

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 9,
                         'axes.edgecolor': '#c3c2b7', 'axes.linewidth': 0.6, 'xtick.color': '#52514e', 'ytick.color': '#0b0b0b',
                         'axes.labelcolor': '#52514e', 'pdf.fonttype': 42})
    COL = {'Profound': '#0d366b', 'Severe': '#1c5cab', 'Moderate': '#3987e5', 'Mild': '#86b6ef'}   # ordinal blue ramp
    TXT = {'Profound': 'white', 'Severe': 'white', 'Moderate': 'white', 'Mild': '#0b0b0b'}
    SURFACE, MUTED, INK = 'white', '#898781', '#0b0b0b'


    def stacked(ax, labels, pct, ns, min_label=6.0, extra=None):
        y = range(len(labels))[::-1]
        left = [0.0] * len(labels)
        for c in CLASSES:
            vals = [p[c] for p in pct]
            ax.barh(list(y), vals, left=left, color=COL[c], edgecolor=SURFACE, linewidth=1.0, height=0.72)
            for yi, l, v in zip(y, left, vals):
                if v >= min_label:
                    ax.text(l + v / 2, yi, f'{v:.0f}', ha='center', va='center', fontsize=7.5, color=TXT[c])
            left = [l + v for l, v in zip(left, vals)]
        ax.set_yticks(list(y)); ax.set_yticklabels(labels)
        ax.set_xlim(0, 100); ax.set_xlabel('Share of gene-disease pairs (%)')
        for yi, n_ in zip(y, ns):
            ax.text(101.5, yi, n_, va='center', ha='left', fontsize=7.5, color='#52514e')
        ax.tick_params(axis='y', length=0)
        for s in ['top', 'right', 'left']:
            ax.spines[s].set_visible(False)


    fig = plt.figure(figsize=(15, 17))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.9, 2.1, 1.15], hspace=0.45, left=0.2, right=0.93, top=0.88, bottom=0.05)
    top = outer[0].subgridspec(1, 2, wspace=0.55)
    bottom = outer[2].subgridspec(1, 2, wspace=0.2, width_ratios=[1.25, 1])
    # A
    axA = fig.add_subplot(top[0, 0])
    axA.bar(CLASSES, A.pairs, color=[COL[c] for c in CLASSES], width=0.62)
    for i, (n_, p_) in enumerate(zip(A.pairs, A.pct)):
        axA.text(i, n_ + A.pairs.max() * 0.02, f'{n_:,}\n({p_}%)', ha='center', va='bottom', fontsize=8.5, color=INK)
    axA.set_ylim(0, A.pairs.max() * 1.25); axA.set_ylabel('Gene-disease pairs')
    axA.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    for s in ['top', 'right']:
        axA.spines[s].set_visible(False)
    axA.yaxis.grid(True, color='#e1e0d9', linewidth=0.5); axA.set_axisbelow(True)
    axA.set_title(f'a  Severity of gene-disease pairs (n = {len(L):,})', loc='left', fontsize=11, fontweight='bold')
    # C
    axC = fig.add_subplot(top[0, 1])
    stacked(axC, list(C.index), [{c: C.at[i, c + '_pct'] for c in CLASSES} for i in C.index], [f'n = {n_:,}' for n_ in C.pairs])
    axC.set_title('c  Severity by mode of inheritance', loc='left', fontsize=11, fontweight='bold')
    # B
    axB = fig.add_subplot(outer[1])
    stacked(axB, list(B.index), [{c: B.at[i, c + '_pct'] for c in CLASSES} for i in B.index], [f'n = {n_:,}' for n_ in B.pairs])
    axB.set_xlabel('Share of gene-disease pairs with a phenotype in the body system (%)')
    axB.set_title('b  Severity by HPO body system (ranked by Profound share)', loc='left', fontsize=11, fontweight='bold')
    # D
    axD = fig.add_subplot(bottom[0, 0])
    dlab = {'All genes (base rate)': 'All genes (base rate)', 'ACMG SF v3.2': 'ACMG secondary findings v3.2',
            'ACMG carrier screening': 'ACMG carrier screening (Tier 3)', "Mackenzie's Mission": "Mackenzie's Mission",
            'Human Intolerome (fetal effect)': 'Human Intolerome (fetal effect)'}
    stacked(axD, [f'{dlab[n]}\n(n = {k:,} genes)' for n, k in zip(D.gene_set, D.genes_labelled)],
            [{c: r[c + '_pct'] for c in CLASSES} for _, r in D.iterrows()],
            [f'{r.Severe_or_Profound_pct:.0f}%' for _, r in D.iterrows()])
    axD.text(101.5, len(D) - 0.45, 'Severe or\nProfound', fontsize=7.5, color='#52514e', va='bottom', ha='left')
    axD.set_xlabel('Share of genes (%)')
    axD.set_title('d  Severity of genes in clinical gene lists', loc='left', fontsize=11, fontweight='bold', pad=16)
    # E
    axE = fig.add_subplot(bottom[0, 1])
    yE = range(len(D))[::-1]
    axE.barh(list(yE), D.early_death_pct, color=[MUTED] + ['#2a78d6'] * (len(D) - 1), height=0.72)
    for yi, v, n_ in zip(yE, D.early_death_pct, D.genes_labelled):
        axE.text(v + 1, yi, f'{v:.0f}%', va='center', fontsize=8.5, color=INK)
    axE.set_yticks(list(yE)); axE.set_yticklabels([]); axE.tick_params(axis='y', length=0)
    axE.set_ylim(axD.get_ylim())
    axE.set_xlim(0, max(D.early_death_pct) * 1.25); axE.set_xlabel('Genes whose condition has an early-death phenotype (%)')
    for s in ['top', 'right', 'left']:
        axE.spines[s].set_visible(False)
    axE.xaxis.grid(True, color='#e1e0d9', linewidth=0.5); axE.set_axisbelow(True)
    axE.set_title('e  Genes with an early-death phenotype (same gene lists)', loc='left', fontsize=11, fontweight='bold', pad=16)
    fig.legend(handles=[Patch(facecolor=COL[c], label=c) for c in CLASSES], loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(0.56, 0.995), fontsize=10, title='Severity class', title_fontsize=10)
    style.save(fig, OUT / 'Figure7.pdf'); style.save(fig, OUT / 'Figure7.png')   # 180 mm wide, 5-8 pt text
    plt.close(fig)

    # ------------------------------------------------------------------ workbook
    notes = [
        f'Pairs: {len(PAIRS_ALL):,} gene-disease pairs in the preprocessing output; {len(PAIRS_ALL) - len(P)} whose strongest GenCC '
        f'classification is Limited are excluded, leaving {len(P):,}; {len(L):,} have a severity class (frequency-filtered version).',
        '7B / Table 2: a pair is counted under every HPO body system that contains one of its phenotypes kept by the frequency filter, '
        'so the rows add up to more than the number of pairs.',
        f'7C: {no_moi} labelled pairs have no inheritance annotation and are not shown. Complex = more than one of AD, AR, X-linked, or other inheritance terms only.',
        '7D: a list gene takes the highest severity among the conditions the list gives for it (matched to dataset pairs exactly, through a '
        'replacement MONDO term, or through a broader or narrower MONDO term). All genes (base rate): highest severity over all pairs of the gene.',
        '7E: early-death phenotype = a phenotype tiered as "Shortened life span: infancy" or "Shortened life span: childhood/adolescence", among phenotypes kept by the frequency filter.',
        'Lists: ACMG SF v3.2 (Miller et al. 2023, doi:10.1016/j.gim.2023.100866), 81 genes; ACMG Tier 3 carrier screening (Gregg et al. 2021, '
        'doi:10.1038/s41436-021-01203-z), 113 genes (97 autosomal recessive, 16 X-linked); Mackenzie\'s Mission (Kirk et al. 2021), '
        f'{mm.gene.nunique():,} genes ({mm_rows - len(mm)} duplicate rows removed); '
        f'Human Intolerome (Yatsenko et al. 2026, doi:10.1016/j.gim.2026.102546), {it_rows} rows, {len(itg)} genes, of which '
        f'{int(itg.in_main_set.sum())} have a fetal effect (Fetal or Maternal,Fetal) with Known or Emerging evidence.',
        'Gene and MIM numbers for ACMG SF and ACMG carrier screening were read from the ACMG-hosted PDFs of the articles (data/gene_lists/*.txt).',
        'Two MIM numbers printed in Gregg et al. 2021 do not match the condition named beside them; the number of the named condition is used (gene_level.py, CS_MIM_NAMED): BCKDHB, maple syrup urine disease, 245600 -> 248600 (Table 2); MCPH1, primary microcephaly 1, 651200 -> 251200 (Table 3). The input file keeps the printed numbers.',
    ]
    with pd.ExcelWriter(OUT / 'figure7_data.xlsx', engine='openpyxl') as xw:
        pd.DataFrame({'notes': notes}).to_excel(xw, sheet_name='README', index=False)
        MS.to_excel(xw, sheet_name='README', index=False, startrow=len(notes) + 2)
        A.reset_index(names='severity').to_excel(xw, sheet_name='7A', index=False)
        B.reset_index().to_excel(xw, sheet_name='7B_Table2', index=False)
        C.reset_index(names='inheritance').to_excel(xw, sheet_name='7C', index=False)
        D.drop(columns=['early_death_genes', 'early_death_pct']).to_excel(xw, sheet_name='7D', index=False)
        D[['gene_set', 'genes_labelled', 'early_death_genes', 'early_death_pct']].to_excel(xw, sheet_name='7E', index=False)
        S.to_excel(xw, sheet_name='Sensitivity', index=False)
        G.to_excel(xw, sheet_name='Gene_severity', index=False)
        for name, df in LISTS.items():
            df.to_excel(xw, sheet_name={'ACMG SF v3.2': 'SF_v3.2', 'ACMG carrier screening': 'CS_Tier3',
                                        "Mackenzie's Mission": 'MM', 'Human Intolerome (fetal effect)': 'Intolerome_fetal'}[name], index=False)
        INTOL_ALL.to_excel(xw, sheet_name='Intolerome_all_rows', index=False)
        from openpyxl.styles import Font
        for ws in xw.book.worksheets:
            for row in ws.iter_rows():
                for c in row:
                    c.font = Font(name='Arial', size=10, bold=(c.row == 1))
            for col in ws.columns:
                width = max(len(str(c.value or '')) for c in col[:300])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
            ws.freeze_panes = 'A2'
    note(f'Wrote {OUT / "Figure7.pdf"}, Figure7.png, figure7_data.xlsx')
    return OUT
