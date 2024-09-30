NUM_SAMPLES = {
    'NASC-TG2': 20000,
    'WHU-RS19': 1005,
    'RSSCN7': 2800,
    'RS_C11': 1232,
    'SIRI-WHU': 2400,
    'NWPU-RESISC45': 31500,
    'PatternNet': 30400,
    'RSD46-WHU': 17516,
    'CLRS': 15000,
    'Optimal-31': 1860,
    'Airbus-Wind-Turbines-Patches': 71504,
    'USTC_SmokeRS': 6225,
    'Satellite-Images-of-Hurricane-Damage': 10000,
    'Million-AID': 10000,
    'UC_Merced_LandUse_MultiLabel': 2100,
    'MLRSNet': 109161,
    'MultiScene': 14000,
    'RSI-CB256': 24747,
    'AID_MultiLabel': 3000,
}

trainer = dict(
    dataset=dict(
        type='VQDatasetRegistry.ConcatDataset',
        name='satin_train',
        num_categories=1,
        datasets=[
            dict(
                type='VQDatasetRegistry.SATINDataset',
                split=split,
                num_val_samples=num_samples // 10,
                train=True,
            ) for split, num_samples in NUM_SAMPLES.items()
        ],
    ),
)
validator = dict(
    dataset=dict(
        type='VQDatasetRegistry.ConcatDataset',
        name='satin_val',
        num_categories=1,
        datasets=[
            dict(
                type='VQDatasetRegistry.SATINDataset',
                split=split,
                num_val_samples=num_samples // 10,
                train=False,
            ) for split, num_samples in NUM_SAMPLES.items()
        ],
    ),
)
