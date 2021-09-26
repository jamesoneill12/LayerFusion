""" remember this is not the same as the settings.py which is the main config.
"""
class Config:
    
    def __init__(self, args):
        
        self.self.args = args
        
    def get_ecoc_method_setting(self, method):
        """first tests ce with codebook so to evaluate with hamming distance and accuracy"""
        """then tests ecoc and also evaluates with hamming distance and accuracy """
        if method == "ss-ecoc":
            losses = ["bce"]  # "ce",
            self.args.codebook = True
            ext = "ecoc/"
            self.args.ss_emb = False
            self.args.cw_mix = False  # when one is on, the other should be off
            # by default cw_mix curriculum is exponential with 0.5 upper threshold
        elif method == "cms-ecoc":
            losses = ["bce"]  # "ce",
            self.args.codebook = True
            ext = "ecoc/"
            self.args.ss_emb = False
            self.args.cw_mix = True  # when one is on, the other should be off
            # by default cw_mix curriculum is exponential with 0.5 upper threshold

        elif method == "ss-ecoc-soft":
            losses = ["bce"]  # "ce",
            self.args.codebook = True
            ext = "ecoc/"
            self.args.ss_emb = True
            self.args.cw_mix = False  # when one is on, the other should be off
            # by default cw_mix curriculum is exponential with 0.5 upper threshold
        elif method == "cms-ecoc-soft":
            losses = ["bce"]  # "ce",
            self.args.codebook = True
            ext = "ecoc/"
            self.args.ss_emb = True
            self.args.cw_mix = True  # when one is on, the other should be off
            # by default cw_mix curriculum is exponential with 0.5 upper threshold
        elif method == "ss-hs":
            losses = ["bce"]  # "ce",
            self.args.codebook = True
            ext = "ecoc/"
            self.args.ss_emb = False
            self.args.cw_mix = False  # when one is on, the other should be off
        elif method == "ss-as":
            losses = ["bce"]  # "ce",
            self.args.codebook = True
            ext = "ecoc/"
            self.args.ss_emb = False
            self.args.cw_mix = False  # when one is on, the other should be off
        
        