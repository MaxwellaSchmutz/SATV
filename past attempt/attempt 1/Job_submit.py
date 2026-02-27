from sf_backtester import BacktestConfig, BacktestRunner, SlurmConfig

slurm_config = SlurmConfig(
    n_cpus=8,
    mem="32G",
    time="03:00:00",
    mail_type="BEGIN,END,FAIL",
    max_concurrent_jobs=30,
)

backtest_config = BacktestConfig(
    signal_name="satv",
    data_path="satv_alphas.parquet",
    gamma=50,
    project_root="/home/msch2022/sf-quant-labs-1",
    byu_email="msch2022@byu.edu",
    constraints=["ZeroBeta", "ZeroInvestment"],
    slurm=slurm_config,
)

backtest_runner = BacktestRunner(backtest_config)
backtest_runner.submit(dry_run=False)
