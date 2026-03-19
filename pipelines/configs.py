from signals.expr import momentum, reversal, bab

configs = [
    {
        'signal_expr': reversal,
        'gamma': 129,
        'constraints': ["ZeroBeta", "ZeroInvestment"]
    },
    {
        'signal_expr': momentum,
        'gamma': 43,
        'constraints': ["ZeroBeta", "ZeroInvestment"]
    },
    {
        'signal_expr': bab,
        'gamma': 37,
        'constraints': ["ZeroInvestment"]
    },
]