from specrsi.mechanisms import frozen_pool,run_pair,run_variant
from specrsi.science.executor import make_scope


def test_real_native_fixed_pool_matches_direct_and_reuses():
    result=run_pair({'mode':'fixed_pool','domain':'reactor','case_ids':[0,1],
                     'candidate_count':3,'budget_ms':300000,'execution_cap_ms':20000,'checker_cap_ms':1000},7)
    assert result['comparison']['common_complete_outcomes_equal']
    assert result['comparison']['same_selected_candidate']
    assert result['comparison']['both_finished_entire_pool']
    reuse=result['variants']['scoped_reuse']
    assert reuse['statistics']['P']==2 and reuse['statistics']['U']==2
    assert reuse['accounting']['checker_wall_seconds']>0
    assert reuse['accounting']['provenance_recheck_wall_seconds']>0
    assert result['pool_before_evaluation'] and result['pool_sha256']
    assert result['learned_heads_used'] is False


def test_budget_envelope_exhaustion_never_scores_partial_candidate():
    scope=make_scope('reactor')
    pool=frozen_pool('reactor',scope,2,0)
    result=run_variant({'mode':'fixed_pool','budget_ms':1,'execution_cap_ms':20000},scope,pool,reuse=True,seed=0)
    assert result['statistics']['complete_candidates']==1
    assert result['selected_id']==pool[0].uid
    assert result['accounting']['charged_search_ms']==0
    assert result['statistics']['stopped_reason'].startswith('insufficient_remaining_budget')
