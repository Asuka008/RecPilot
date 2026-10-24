"""Pure-Python regression checks for frozen validation selection."""
import copy
import unittest
from scripts.hybrid_test_admission import selected_alpha

class HybridAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.report={'status':'PASS','split':'val','alphas':[0,.1,.25,.5],
                     'selection':{'alpha':.25,'status':'candidate_for_review'},'variants':{}}
        for alpha,score,tail in [(0,.038,0),(.1,.041,.001),(.25,.044,.01),(.5,.036,.04)]:
            groups={'all':{'users':100,'metrics':{'NDCG@10':score,'Recall@20':.1}},
                    '1-5':{'users':10,'metrics':{'NDCG@10':tail,'Recall@20':tail}}}
            self.report['variants'][str(alpha)]={'split':'val','groups_by_target_train_frequency':groups,
                                               'groups_by_history_length':copy.deepcopy(groups)}

    def test_predeclared_winner(self):
        self.assertEqual(selected_alpha(self.report),.25)

    def test_test_metrics_failed_runs_and_wrong_selection_rejected(self):
        for field,value in [('split','test'),('status','FAIL'),('alphas',[0,.25])]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                selected_alpha(dict(self.report,**{field:value}))
        self.report['selection']['alpha']=.5
        with self.assertRaises(ValueError):selected_alpha(self.report)

    def test_nonfinite_or_changed_user_groups_rejected(self):
        for field,value in [('users',99),('metrics',{'NDCG@10':float('nan')})]:
            report=copy.deepcopy(self.report)
            report['variants']['0.25']['groups_by_target_train_frequency']['all'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):selected_alpha(report)

    def test_ties_choose_smaller_weight(self):
        self.report['variants']['0.1']['groups_by_target_train_frequency']['all']['metrics']['NDCG@10']=.044
        self.report['selection']['alpha']=.1
        self.assertEqual(selected_alpha(self.report),.1)

if __name__=='__main__':unittest.main()
