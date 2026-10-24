"""Small NumPy checks of score scaling and fallback semantics."""
import unittest
import numpy as np
from scripts.evaluate_hybrid_validation import combine

class HybridTest(unittest.TestCase):
    def test_zero_weight_and_no_evidence_preserve_original_scores(self):
        a=np.array([0.,-.2,.7])
        np.testing.assert_array_equal(combine(a,[0,2,1],0),a)
        np.testing.assert_array_equal(combine(a,[0,0,0],.5),a)

    def test_cf_scale_does_not_change_blend(self):
        a=[0,-.2,.8];b=np.array([0,3,1])
        np.testing.assert_allclose(combine(a,b,.25),combine(a,7*b,.25))
        np.testing.assert_allclose(combine(a,b,.25),[.375,.55,.7583333333333333])

    def test_bad_inputs_are_rejected(self):
        for a,b,w in [([0,1],[0,-1],.2),([0,1],[0],.2),([0,float('nan')],[0,1],.2),([0,1],[0,1],2)]:
            with self.assertRaises(ValueError): combine(a,b,w)

if __name__=='__main__': unittest.main()
