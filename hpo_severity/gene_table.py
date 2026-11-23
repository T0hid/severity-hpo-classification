"""
Gene-disease-phenotype table with a severity class for every gene-disease pair (Figure 7, Table 2).

Run through the entry point (python run.py figures or python run.py verify). Reads the files in
data/inputs/ (unpacked from data/ontologies.zip and data/gene_disease_sources.zip) and the agent
classification file. Writes one row per gene x disease x HPO phenotype. A pair with no phenotype
annotation has one row with empty phenotype fields. Pair-level fields repeat on every row of the pair.

Steps
 1. MONDO (release 2026-09-01): active disease terms, their labels and skos:exactMatch cross-references to OMIM,
    Orphanet and DECIPHER; gene-disease pairs from the germline relationships RO:0004003 ("has material basis in
    germline mutation in") and RO:0004001 ("has material basis in gain of function germline mutation in"),
    asserted as edges or in logical definitions, on active terms.
 2. HPO (release 2026-09-01, annotation version 2026-09-02): positive annotations (empty qualifier, aspect P);
    the source disease identifier is mapped to MONDO only when exactly one active MONDO term has it as exactMatch.
    No ancestor or descendant expansion.
 3. GenCC: pairs with at least one Definitive, Strong or Moderate submission for the exact HGNC x MONDO pair are
    added. Removed: GenCC-added pairs on MONDO terms that are obsolete or absent in MONDO 2026-09-01; GenCC-added
    pairs on diseases linked to more than UMBRELLA_MAX_GENES genes; pairs whose strongest GenCC classification is
    Disputed, Refuted or No Known Disease Relationship. Pairs whose strongest classification is Limited are flagged.
 4. Inheritance: HPO inheritance annotations (aspect I) of the disease; if none, GenCC mode of inheritance
    (Definitive/Strong/Moderate submissions) for the exact pair.
 5. Frequency: a phenotype whose every source annotation gives frequency 0 is recorded as absent and removed from
    both severity versions. Otherwise it is kept for the frequency-filtered version if no source gives a frequency,
    or if the highest given frequency is at least FREQ_CUTOFF.
 6. Tiers: each HPO term gets the agent's tier (1, 2, 3, 4 or NFC) and category from the agent classification file
    (published_run/agent_classifications.zip by default).
 7. Broader terms: within a pair, a tier 1-3 term is not counted when a more specific counted term of the same
    pair has the same tier (descendant through is_a in hp.obo, any number of levels).
 8. Severity per pair from the distinct counted tier 1, 2, 3 terms: Profound T1>1; Severe T1=1 or (T2>=1 and
    T2+T3>=4); Moderate T2>=1 or T3>=1; Mild otherwise. Pairs whose labelled terms are all Out of Scope/NFC
    are excluded. Two versions: _freq (frequency-filtered) and _all (no frequency filter; absent terms removed).
Reference gene lists (SF, CS, MM, Intolerome) are matched in gene_level.py, not here.
"""
import os, re, json, hashlib, collections, datetime, pathlib
import pandas as pd

from . import paths


def build(output_file, classifications=None):
    """Builds the table and writes it to output_file. Returns the output path."""
    ROOT = paths.ROOT
    OUT_FILE = pathlib.Path(output_file)
    OUT_DIR = OUT_FILE.parent
    UMBRELLA_MAX_GENES = 10
    FREQ_CUTOFF = 0.30

    IN = {
        'mondo_json': paths.MONDO,                  # MONDO 2026-09-01 release
        'hpoa': paths.HPO_ANNOTATIONS,              # HPO 2026-09-01 release, annotation version 2026-09-02
        'hp_obo': paths.HPO_2026,                   # HPO 2026-09-01 release
        'tiers': pathlib.Path(classifications or paths.CLASSIFICATIONS),   # agent tier and category per HPO term
        'gencc': paths.GENCC,                       # GenCC submissions export of 15 November 2026
    }


    def note(msg=''):
        print(msg, flush=True)


    def sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        return h.hexdigest()


    note(f'preprocessing.py run {datetime.datetime.now().isoformat(timespec="seconds")}')
    note(f'Settings: UMBRELLA_MAX_GENES={UMBRELLA_MAX_GENES}, FREQ_CUTOFF={FREQ_CUTOFF}')
    for k, p in IN.items():
        note(f'  input {p.name:28s} sha256 {sha(p)}')

    # ------------------------------------------------------------------ 1. MONDO
    def norm(v):
        v = str(v).strip().replace('Orphanet:', 'ORPHA:')
        if re.fullmatch(r'(OMIM|ORPHA|DECIPHER):\d+', v):
            return v
        for pat, prefix in [(r'omim.org/entry/(\d+)$', 'OMIM'), (r'identifiers.org/omim/(\d+)$', 'OMIM'),
                            (r'[/_]Orphanet_(\d+)$', 'ORPHA'), (r'identifiers.org/orphanet/(\d+)$', 'ORPHA')]:
            m = re.search(pat, v)
            if m:
                return prefix + ':' + m.group(1)
        return None


    def mondo_id(iri):
        return iri.rsplit('/', 1)[-1].replace('MONDO_', 'MONDO:')


    def hgnc_id(iri):
        return 'HGNC:' + iri.rsplit('/', 1)[-1]


    g = json.loads(IN['mondo_json'].read_text(encoding='utf-8'))['graphs'][0]
    src2mondo, mondo2src, mondo_label, gene_label = collections.defaultdict(set), collections.defaultdict(set), {}, {}
    n_disease_classes = 0
    for n in g['nodes']:
        if 'identifiers.org/hgnc/' in n['id']:
            gene_label[hgnc_id(n['id'])] = n.get('lbl', '')
        if '/MONDO_' not in n['id'] or n.get('meta', {}).get('deprecated'):
            continue
        mid = mondo_id(n['id'])
        mondo_label[mid] = n.get('lbl', '')
        n_disease_classes += n.get('type') == 'CLASS'
        for b in n.get('meta', {}).get('basicPropertyValues', []):
            if b['pred'] == 'http://www.w3.org/2004/02/skos/core#exactMatch':
                sid = norm(b['val'])
                if sid:
                    src2mondo[sid].add(mid); mondo2src[mid].add(sid)
    GERMLINE = {'http://purl.obolibrary.org/obo/RO_0004003', 'http://purl.obolibrary.org/obo/RO_0004001'}
    germ = set()
    for e in g['edges']:
        if e['pred'] in GERMLINE and 'identifiers.org/hgnc/' in e['obj'] and '/MONDO_' in e['sub']:
            germ.add((hgnc_id(e['obj']), mondo_id(e['sub'])))
    for a in g.get('logicalDefinitionAxioms', []):
        if '/MONDO_' not in a.get('definedClassId', ''):
            continue
        for r in a.get('restrictions', []):
            if r.get('propertyId') in GERMLINE and 'identifiers.org/hgnc/' in r.get('fillerId', ''):
                germ.add((hgnc_id(r['fillerId']), mondo_id(a['definedClassId'])))
    del g
    germ = sorted((m, h) for h, m in germ if m in mondo_label)          # active MONDO terms only
    mp = pd.DataFrame([(h, gene_label.get(h, ''), m, mondo_label[m]) for m, h in germ],
                      columns=['HGNC', 'Gene Name', 'MONDO_ID', 'Disease'])
    assert not mp.duplicated(['HGNC', 'MONDO_ID']).any()
    note(f'MONDO: {n_disease_classes:,} active disease terms; germline gene-disease pairs {len(mp):,} '
         f'({mp.HGNC.nunique():,} genes, {mp.MONDO_ID.nunique():,} diseases)')


    def to_mondo(sid):
        s = src2mondo.get(sid, set())
        return next(iter(s)) if len(s) == 1 else None

    # ------------------------------------------------------------------ 2. HPO
    hp_name, cur = {}, None
    for line in IN['hp_obo'].read_text(encoding='utf-8').splitlines():
        if line.startswith('id: HP:'): cur = line[4:]
        elif line.startswith('name: ') and cur: hp_name[cur] = line[6:]; cur = None
    parents, cur = collections.defaultdict(set), None                  # is_a links, [Term] stanzas only
    for line in IN['hp_obo'].read_text(encoding='utf-8').splitlines():
        if line.startswith('['):
            cur = '' if line == '[Term]' else None
        elif cur == '' and line.startswith('id: '):
            cur = line[4:].strip()
        elif cur and line.startswith('is_a: '):
            parents[cur].add(line[6:].split('!')[0].strip())
    children = collections.defaultdict(set)
    for c, ps in parents.items():
        for p in ps:
            children[p].add(c)


    def closure(t, links, memo):
        if t not in memo:
            seen, stack = set(), list(links.get(t, ()))
            while stack:
                x = stack.pop()
                if x not in seen:
                    seen.add(x); stack.extend(links.get(x, ()))
            memo[t] = seen
        return memo[t]


    _anc, _des = {}, {}
    ancestors = lambda t: closure(t, parents, _anc)
    descendants = lambda t: closure(t, children, _des)

    hpoa = pd.read_csv(IN['hpoa'], sep='\t', comment='#', dtype=str, keep_default_na=False)
    src_ids = set(hpoa.database_id)
    note(f'HPO annotations: {len(hpoa):,} rows; source disease identifiers {len(src_ids):,}, mapped to exactly one MONDO term '
         f'{sum(1 for s in src_ids if to_mondo(norm(s) or s)):,}')
    hpoa['MONDO_ID'] = hpoa.database_id.map(lambda s: to_mondo(norm(s) or s))
    pos = hpoa[(hpoa.qualifier == '') & (hpoa.aspect == 'P')]
    ev = pos.dropna(subset=['MONDO_ID']).rename(columns={'hpo_id': 'HPO_ID', 'database_id': 'source_disease_id'})
    ev = ev[['MONDO_ID', 'HPO_ID', 'source_disease_id', 'frequency']].copy()
    ev['frequency'] = ev.frequency.where(ev.frequency != '', None)
    note(f'Positive phenotype annotations: {len(pos):,}; mapped to MONDO {len(ev):,}')
    pdis = ev.groupby(['MONDO_ID', 'HPO_ID']).source_disease_id.apply(lambda s: '|'.join(sorted(set(s)))).reset_index(name='source_disease_ids')
    pdis['HPO_Term'] = pdis.HPO_ID.map(hp_name)
    assert pdis.HPO_Term.notna().all(), 'annotated HPO term not in hp.obo'
    note(f'Disease-phenotype pairs: {len(pdis):,} on {pdis.MONDO_ID.nunique():,} diseases')

    # ------------------------------------------------------------------ 6. tiers (read early, used below)
    lab = pd.read_csv(IN['tiers'], dtype=str, usecols=['hpo_id', 'classification_tier', 'classification_category'])
    lab['classification_category'] = lab.classification_category.fillna('').str.replace('â€“', '–').str.strip()
    lab['classification_tier'] = lab.classification_tier.fillna('').str.strip()
    assert lab.hpo_id.is_unique, 'tier file has a repeated hpo_id'
    assert set(lab.classification_tier) <= {'1', '2', '3', '4', 'NFC'}, 'unexpected tier value'
    lab = lab.set_index('hpo_id')
    note(f'Tiers: {len(lab):,} terms; {lab.classification_tier.value_counts().to_dict()}')

    # ------------------------------------------------------------------ 3. GenCC per exact pair, and the pair set
    RANK = {'Definitive': 1, 'Strong': 2, 'Moderate': 3, 'Supportive': 4, 'Limited': 5,
            'Disputed Evidence': 6, 'Refuted Evidence': 7, 'No Known Disease Relationship': 8}
    gc = pd.read_csv(IN['gencc'], dtype=str, usecols=['gene_curie', 'gene_symbol', 'disease_curie', 'classification_title', 'moi_title', 'submitter_title'])
    assert gc.classification_title.isin(RANK).all(), 'unknown GenCC classification'
    gc['rank'] = gc.classification_title.map(RANK)
    gsum = gc.groupby(['gene_curie', 'disease_curie']).agg(
        gencc_best_rank=('rank', 'min'),
        gencc_classifications=('classification_title', lambda s: '|'.join(sorted(set(s), key=RANK.get))),
        gencc_n_submitters=('submitter_title', 'nunique'),
        gencc_any_contradicting=('rank', lambda s: bool((s >= 6).any())),
        gencc_moi_dsm=('moi_title', lambda s: '|'.join(sorted(set(s.dropna())))),
    ).reset_index().rename(columns={'gene_curie': 'HGNC', 'disease_curie': 'MONDO_ID'})
    dsm_moi = gc[gc['rank'] <= 3].groupby(['gene_curie', 'disease_curie']).moi_title.apply(lambda s: '|'.join(sorted(set(s.dropna()))))
    gsum['gencc_moi_dsm'] = [dsm_moi.get((a, b), '') for a, b in zip(gsum.HGNC, gsum.MONDO_ID)]
    gsum['gencc_best'] = gsum.gencc_best_rank.map({v: k for k, v in RANK.items()})

    mp['pair_source'] = 'MONDO_germline'
    known = set(zip(mp.HGNC, mp.MONDO_ID))
    not_in_mondo = pd.Series([(a, b) not in known for a, b in zip(gsum.HGNC, gsum.MONDO_ID)], index=gsum.index)
    add = gsum[(gsum.gencc_best_rank <= 3) & not_in_mondo]
    sym = gc.drop_duplicates('gene_curie').set_index('gene_curie').gene_symbol
    add = pd.DataFrame({'HGNC': add.HGNC, 'Gene Name': add.HGNC.map(sym), 'MONDO_ID': add.MONDO_ID,
                        'Disease': add.MONDO_ID.map(mondo_label), 'pair_source': 'GenCC_DSM_only'})
    note(f'GenCC Definitive/Strong/Moderate pairs not in MONDO: {len(add):,}')
    inactive = add.Disease.isna()
    note(f'  on MONDO terms obsolete or absent in MONDO 2026-09-01 (dropped): {int(inactive.sum())}')
    add = add[~inactive]
    pairs = pd.concat([mp, add], ignore_index=True)
    pairs = pairs.merge(gsum.drop(columns='gencc_best_rank'), on=['HGNC', 'MONDO_ID'], how='left')
    pairs['gencc_best'] = pairs.gencc_best.fillna('Not in GenCC')
    pairs['gencc_any_contradicting'] = pairs.gencc_any_contradicting.eq(True)
    for c in ['gencc_classifications', 'gencc_moi_dsm']:
        pairs[c] = pairs[c].fillna('')
    pairs['gencc_n_submitters'] = pairs.gencc_n_submitters.fillna(0).astype(int)

    genes_per_dis = pairs.groupby('MONDO_ID').HGNC.nunique()
    pairs['genes_linked_to_disease_before_rules'] = pairs.MONDO_ID.map(genes_per_dis)
    umb = (pairs.pair_source == 'GenCC_DSM_only') & (pairs.genes_linked_to_disease_before_rules > UMBRELLA_MAX_GENES)
    contra = pairs.gencc_best.isin(['Disputed Evidence', 'Refuted Evidence', 'No Known Disease Relationship'])
    note(f'Removed, GenCC-added pairs on diseases with more than {UMBRELLA_MAX_GENES} genes: {int(umb.sum()):,} pairs on {pairs[umb].MONDO_ID.nunique()} diseases')
    note(f'Removed, strongest GenCC classification Disputed/Refuted/No Known: {int(contra.sum())} pairs')
    pairs = pairs[~umb & ~contra].reset_index(drop=True)
    pairs['flag_limited_only'] = pairs.gencc_best == 'Limited'
    pairs['OMIM_IDs'] = pairs.MONDO_ID.map(lambda m: '|'.join(sorted(x for x in mondo2src.get(m, ()) if x.startswith('OMIM:'))))
    pairs['ORPHA_IDs'] = pairs.MONDO_ID.map(lambda m: '|'.join(sorted(x for x in mondo2src.get(m, ()) if x.startswith('ORPHA:'))))
    note(f'Pairs kept: {len(pairs):,} ({pairs.HGNC.nunique():,} genes, {pairs.MONDO_ID.nunique():,} diseases); '
         f'sources {pairs.pair_source.value_counts().to_dict()}; flagged Limited-only {int(pairs.flag_limited_only.sum())}')

    # ------------------------------------------------------------------ 4. inheritance
    MOI_GROUP_HPO = {'HP:0000007': 'AR', 'HP:0000006': 'AD', 'HP:0001417': 'XL', 'HP:0001419': 'XL', 'HP:0001423': 'XL'}
    MOI_GROUP_GENCC = {'Autosomal recessive': 'AR', 'Autosomal dominant': 'AD', 'X-linked': 'XL', 'X-linked recessive': 'XL'}
    inh = hpoa[(hpoa.aspect == 'I') & (hpoa.qualifier == '')].dropna(subset=['MONDO_ID'])
    moi_terms = inh.groupby('MONDO_ID').hpo_id.apply(lambda s: sorted(set(s))).to_dict()


    def group(labels):
        main = {x for x in labels if x in ('AR', 'AD', 'XL')}
        if len(main) == 1: return next(iter(main))
        if len(main) > 1: return 'Multiple'
        return 'Other' if labels else ''


    moi_g, moi_src, moi_txt = [], [], []
    for m_, gm in zip(pairs.MONDO_ID, pairs.gencc_moi_dsm):
        t = moi_terms.get(m_, [])
        if t:
            moi_g.append(group({MOI_GROUP_HPO.get(x, 'Other') for x in t})); moi_src.append('HPO'); moi_txt.append('|'.join(hp_name.get(x, x) for x in t))
        else:
            gl = [x for x in gm.split('|') if x and x != 'Unknown']
            if gl:
                moi_g.append(group({MOI_GROUP_GENCC.get(x, 'Other') for x in gl})); moi_src.append('GenCC'); moi_txt.append('|'.join(gl))
            else:
                moi_g.append('Not annotated'); moi_src.append(''); moi_txt.append('')
    pairs['MOI_group'], pairs['MOI_source'], pairs['MOI_detail'] = moi_g, moi_src, moi_txt
    note(f'Inheritance: {pairs.MOI_group.value_counts().to_dict()}; source {pairs.MOI_source.replace("", "none").value_counts().to_dict()}')

    # ------------------------------------------------------------------ 5. phenotype rows and frequency
    tri = pairs[['HGNC', 'MONDO_ID']].merge(pdis[['HPO_ID', 'HPO_Term', 'MONDO_ID', 'source_disease_ids']], on='MONDO_ID')
    tri['agent_tier'] = tri.HPO_ID.map(lab.classification_tier).fillna('')
    tri['agent_category'] = tri.HPO_ID.map(lab.classification_category).fillna('')
    tri['in_paper_term_set'] = tri.HPO_ID.isin(lab.index)

    FREQ_TERM = {'HP:0040280': 1.0, 'HP:0040281': 0.9, 'HP:0040282': 0.55, 'HP:0040283': 0.17, 'HP:0040284': 0.025, 'HP:0040285': 0.0}


    def fval(v):
        if not isinstance(v, str) or not v.strip(): return None
        v = v.strip()
        if v in FREQ_TERM: return FREQ_TERM[v]
        m = re.fullmatch(r'(\d+)\s*/\s*(\d+)', v)
        if m: return int(m.group(1)) / int(m.group(2)) if int(m.group(2)) > 0 else None
        m = re.fullmatch(r'([\d.]+)\s*%', v)
        if m: return float(m.group(1)) / 100
        raise ValueError(f'unparsable frequency {v!r}')


    ev['f'] = ev.frequency.map(fval)
    fq = ev.groupby(['MONDO_ID', 'HPO_ID']).agg(n_source_annotations=('f', 'size'), n_with_frequency=('f', 'count'),
                                                 max_frequency=('f', 'max'), frequencies=('frequency', lambda s: '|'.join(sorted(set(s.dropna()))))).reset_index()
    every = fq.n_with_frequency == fq.n_source_annotations
    some = (fq.n_with_frequency > 0) & ~every
    fq['recorded_absent'] = every & (fq.max_frequency == 0)
    fq['freq_kept'] = ~fq.recorded_absent & (fq.max_frequency.isna() | (fq.max_frequency >= FREQ_CUTOFF))
    fq['freq_status'] = 'no_frequency'
    fq.loc[some & (fq.max_frequency >= FREQ_CUTOFF), 'freq_status'] = 'some_frequency_kept'
    fq.loc[some & (fq.max_frequency < FREQ_CUTOFF), 'freq_status'] = 'some_frequency_lt30_removed'
    fq.loc[every & (fq.max_frequency >= FREQ_CUTOFF), 'freq_status'] = 'all_frequency_ge30_kept'
    fq.loc[every & (fq.max_frequency < FREQ_CUTOFF), 'freq_status'] = 'all_frequency_lt30_removed'
    fq.loc[fq.recorded_absent, 'freq_status'] = 'recorded_absent_removed'
    tri = tri.merge(fq[['MONDO_ID', 'HPO_ID', 'frequencies', 'max_frequency', 'freq_status', 'recorded_absent', 'freq_kept']], on=['MONDO_ID', 'HPO_ID'], how='left')
    assert tri.freq_kept.notna().all(), 'phenotype row without evidence record'
    tri['freq_kept'] = tri.freq_kept.astype(bool)
    tri['recorded_absent'] = tri.recorded_absent.astype(bool)
    assert not (tri.recorded_absent & tri.freq_kept).any(), 'absent phenotype kept by frequency filter'
    note(f'Phenotype rows: {len(tri):,}; with a tier {int(tri.in_paper_term_set.sum()):,}; recorded absent {int(tri.recorded_absent.sum()):,} rows in '
         f'{tri[tri.recorded_absent][["HGNC", "MONDO_ID"]].drop_duplicates().shape[0]:,} pairs; '
         f'frequency status {tri.freq_status.value_counts().to_dict()}')

    # ------------------------------------------------------------------ 7. broader same-tier terms
    counted = {'all': tri.in_paper_term_set & tri.agent_tier.isin(['1', '2', '3']) & ~tri.recorded_absent}
    counted['freq'] = counted['all'] & tri.freq_kept


    def drop_by_ancestors(mask):
        drop = pd.Series(False, index=tri.index)
        for _, grp in tri.loc[mask, ['HGNC', 'MONDO_ID', 'HPO_ID', 'agent_tier']].groupby(['HGNC', 'MONDO_ID', 'agent_tier']):
            if len(grp) < 2: continue
            anc = set().union(*(ancestors(h) for h in grp.HPO_ID))
            for idx, h in zip(grp.index, grp.HPO_ID):
                if h in anc: drop[idx] = True
        return drop


    def drop_by_descendants(mask):           # second, independent implementation
        drop = pd.Series(False, index=tri.index)
        for _, grp in tri.loc[mask, ['HGNC', 'MONDO_ID', 'HPO_ID', 'agent_tier']].groupby(['HGNC', 'MONDO_ID', 'agent_tier']):
            hs = set(grp.HPO_ID)
            for idx, h in zip(grp.index, grp.HPO_ID):
                if descendants(h) & (hs - {h}): drop[idx] = True
        return drop


    known_terms = set(parents) | {p for ps in parents.values() for p in ps}
    assert set(tri.loc[counted['all'], 'HPO_ID']) <= known_terms, 'counted term missing from hp.obo'
    for v in ['all', 'freq']:
        d1, d2 = drop_by_ancestors(counted[v]), drop_by_descendants(counted[v])
        assert d1.equals(d2), f'broader-term implementations disagree ({v})'
        tri['dropped_broader_same_tier_' + v] = d1
        note(f'Broader same-tier terms not counted (_{v}): {int(d1.sum()):,} rows in '
             f'{tri[d1][["HGNC", "MONDO_ID"]].drop_duplicates().shape[0]:,} pairs; by tier {tri.loc[d1, "agent_tier"].value_counts().to_dict()}')

    # ------------------------------------------------------------------ 8. severity per pair
    def sev(t1, t2, t3):
        if t1 > 1: return 'Profound'
        if t1 == 1: return 'Severe'
        if t2 >= 1: return 'Severe' if t2 + t3 >= 4 else 'Moderate'
        if t3 >= 1: return 'Moderate'
        return 'Mild'


    def summarise(rows, sfx):
        L = rows[rows.in_paper_term_set].copy()
        L['counted_tier'] = L.agent_tier.where(L.agent_tier.isin(['1', '2', '3']) & ~L['dropped_broader_same_tier' + sfx], '')
        c = L.groupby(['HGNC', 'MONDO_ID']).agg(
            T1=('counted_tier', lambda s: int((s == '1').sum())), T2=('counted_tier', lambda s: int((s == '2').sum())),
            T3=('counted_tier', lambda s: int((s == '3').sum())), T4=('agent_tier', lambda s: int((s == '4').sum())),
            NFC=('agent_tier', lambda s: int((s == 'NFC').sum())),
            CS=('agent_category', lambda s: int(s.str.startswith('Clinical Sign').sum())),
            labelled=('agent_tier', 'size')).reset_index()
        c['status'] = ['excluded_nfc_only' if lbl == nfc else 'labelled' for lbl, nfc in zip(c.labelled, c.NFC)]
        c['severity'] = [sev(a, b, d) if s == 'labelled' else '' for a, b, d, s in zip(c.T1, c.T2, c.T3, c.status)]
        c = c.rename(columns={'T1': 'n_T1', 'T2': 'n_T2', 'T3': 'n_T3', 'T4': 'n_T4_incl_clinical_sign', 'NFC': 'n_out_of_scope_or_NFC',
                              'CS': 'n_clinical_sign', 'labelled': 'n_labelled_terms'})
        return c.set_index(['HGNC', 'MONDO_ID']).add_suffix(sfx).reset_index()


    pairs = pairs.merge(summarise(tri[~tri.recorded_absent], '_all'), on=['HGNC', 'MONDO_ID'], how='left') \
                 .merge(summarise(tri[tri.freq_kept], '_freq'), on=['HGNC', 'MONDO_ID'], how='left')
    nph = tri.groupby(['HGNC', 'MONDO_ID']).agg(n_phenotypes=('HPO_ID', 'size'), n_phenotypes_recorded_absent=('recorded_absent', 'sum'),
                                                 n_phenotypes_after_frequency_filter=('freq_kept', 'sum')).reset_index()
    pairs = pairs.merge(nph, on=['HGNC', 'MONDO_ID'], how='left')
    for c in ['n_phenotypes', 'n_phenotypes_after_frequency_filter', 'n_phenotypes_recorded_absent']:
        pairs[c] = pairs[c].fillna(0).astype(int)
    for sfx in ['_all', '_freq']:
        st = 'status' + sfx
        pairs[st] = pairs[st].fillna('no_labelled_terms')
        if sfx == '_freq':
            pairs.loc[(pairs.n_phenotypes > 0) & (pairs.n_phenotypes_after_frequency_filter == 0), st] = 'no_phenotypes_after_frequency_filter'
        else:
            pairs.loc[(pairs.n_phenotypes > 0) & (pairs.n_phenotypes == pairs.n_phenotypes_recorded_absent), st] = 'only_absent_phenotypes'
        pairs.loc[pairs.n_phenotypes == 0, st] = 'no_phenotypes'
        pairs['severity' + sfx] = pairs['severity' + sfx].fillna('')
        for c in [c for c in pairs.columns if c.startswith('n_') and c.endswith(sfx)]:
            pairs[c] = pairs[c].fillna(0).astype(int)
        note(f'Severity{sfx}: {pairs[st].value_counts().to_dict()}; '
             f'classes {pairs.loc[pairs[st] == "labelled", "severity" + sfx].value_counts().reindex(["Profound", "Severe", "Moderate", "Mild"]).to_dict()}')

    # ------------------------------------------------------------------ assemble
    pair_cols = ['HGNC', 'Gene Name', 'MONDO_ID', 'Disease', 'pair_source', 'OMIM_IDs', 'ORPHA_IDs',
                 'MOI_group', 'MOI_source', 'MOI_detail',
                 'gencc_best', 'gencc_classifications', 'gencc_n_submitters', 'gencc_any_contradicting', 'flag_limited_only',
                 'n_phenotypes', 'n_phenotypes_recorded_absent', 'n_phenotypes_after_frequency_filter', 'status_freq', 'severity_freq', 'n_T1_freq', 'n_T2_freq', 'n_T3_freq',
                 'n_T4_incl_clinical_sign_freq', 'n_clinical_sign_freq', 'n_out_of_scope_or_NFC_freq', 'n_labelled_terms_freq',
                 'status_all', 'severity_all', 'n_T1_all', 'n_T2_all', 'n_T3_all',
                 'n_T4_incl_clinical_sign_all', 'n_clinical_sign_all', 'n_out_of_scope_or_NFC_all', 'n_labelled_terms_all']
    term_cols = ['HPO_ID', 'HPO_Term', 'source_disease_ids', 'in_paper_term_set', 'agent_tier', 'agent_category',
                 'frequencies', 'max_frequency', 'freq_status', 'recorded_absent', 'freq_kept',
                 'dropped_broader_same_tier_all', 'dropped_broader_same_tier_freq']
    master = pairs[pair_cols].merge(tri[['HGNC', 'MONDO_ID'] + term_cols], on=['HGNC', 'MONDO_ID'], how='left')
    master = master.rename(columns={'Gene Name': 'Gene_symbol'}).sort_values(['Gene_symbol', 'MONDO_ID', 'HPO_ID'], na_position='first').reset_index(drop=True)
    for c in ['dropped_broader_same_tier_all', 'dropped_broader_same_tier_freq']:
        master[c] = master[c].eq(True)

    # ------------------------------------------------------------------ checks
    rows = master.dropna(subset=['HPO_ID'])
    assert not rows.duplicated(['HGNC', 'MONDO_ID', 'HPO_ID']).any(), 'duplicate gene-disease-phenotype rows'
    assert master[['HGNC', 'MONDO_ID']].drop_duplicates().shape[0] == len(pairs), 'pair count mismatch'
    assert master.loc[master.pair_source == 'GenCC_DSM_only', 'gencc_best'].isin(['Definitive', 'Strong', 'Moderate']).all()
    gpd = pairs.groupby('MONDO_ID').HGNC.nunique()
    assert not ((pairs.pair_source == 'GenCC_DSM_only') & (pairs.MONDO_ID.map(gpd) > UMBRELLA_MAX_GENES)).any()
    assert not pairs.gencc_best.isin(['Disputed Evidence', 'Refuted Evidence', 'No Known Disease Relationship']).any()
    lbl = rows[rows.in_paper_term_set.astype(bool)]
    assert (lbl.agent_tier.values == lab.loc[lbl.HPO_ID, 'classification_tier'].values).all(), 'tier join error'
    assert not (rows.dropped_broader_same_tier_all & rows.recorded_absent.astype(bool)).any()
    assert not (rows.dropped_broader_same_tier_freq & ~rows.freq_kept.astype(bool)).any()
    P = pairs.set_index(['HGNC', 'MONDO_ID'])
    for v, keep in [('all', ~rows.recorded_absent.astype(bool)), ('freq', rows.freq_kept.astype(bool))]:
        use = keep & rows.in_paper_term_set.astype(bool) & rows.agent_tier.isin(['1', '2', '3']) & ~rows['dropped_broader_same_tier_' + v]
        rc = rows[use].groupby(['HGNC', 'MONDO_ID']).agent_tier.value_counts().unstack(fill_value=0).reindex(P.index, fill_value=0)
        for t in ['1', '2', '3']:
            assert (rc.get(t, 0).astype(int).values == P[f'n_T{t}_{v}'].values).all(), f'n_T{t}_{v} not reproducible from rows'
        re_ = [sev(a, b, c) if s == 'labelled' else '' for a, b, c, s in zip(P[f'n_T1_{v}'], P[f'n_T2_{v}'], P[f'n_T3_{v}'], P['status_' + v])]
        assert (pd.Series(re_, index=P.index) == P['severity_' + v]).all(), f'severity_{v} not reproducible from counts'
    note(f'Checks passed. Output rows {len(master):,}; columns {master.shape[1]}; pairs {len(pairs):,}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
    note(f'Wrote {OUT_FILE}')
    return OUT_FILE
