# Example of training a SpatioTemporalModel on HPC

## Folder structure

- example_training.py: example training script
- example.slurm: example SLURM script to execute the training script on SLURM system
- eso4clima_24438134_subset.out: example SLURM job output file of an execution on a subset of the global dataset. The dataset has two years of data (2020-2021) and the spatial coverage is from 30S to 30N and from 30W to 30E.
- eso4clima_24449471_full.out: example SLURM job output file of an execution on the full dataset, two years of data (2020-2021) and almost global coverage (from 80S to 80N and from 179.99W to 179.99E). The training only executed for 1 hour and cuted off by SLURM time limit.

## Execute training tasks on SLURM system

1. Make a working directory

```sh
mkdir training
cd training
```

2. Clone this repo
```sh
git clone git@github.com:ESMValGroup/ClimaNet.git
```

3. Install uv for dependency management. Se [uv doc](https://docs.astral.sh/uv/getting-started/installation/).

4. Create a venv and install Python dependencies using uv
```sh
cd ClimaNet
```

```
uv sync
```

A `.venv` dir will appear

5. Copy the python script and slurm script into the working dir:

```sh
cp ClimaNet/scripts/example* .
```

6. Config `example.slurm`, in the `source ...` line, make sure the venv just created is activated.
   Note that the account is the ESO4CLIMA project account, which is shared by multiple users.

7. Config `example.py`, make sure the path of input data and land mask data is correct.

8. Execute the SLURM job
```sh
sbatch example.slurm
```

## Check the efficiency of resource usage

In the SLURM job output, you can find the line like this:

```
==== Slurm accounting summary 23743544 ====
JobID|NTasks|AveCPU|AveRSS|MaxRSS|MaxVMSize|TRESUsageInAve|TRESUsageInMax
23743544.extern|1|00:00:00|856K|3752K|641376K|cpu=00:00:00,energy=0,fs/disk=2332,mem=856K,pages=2,vmem=217160K|cpu=00:00:00,energy=0,fs/disk=2332,mem=3752K,pages=2,vmem=641376K
23743544.batch|1|04:21:01|11964K|4102096K|37743716K|cpu=04:21:01,energy=0,fs/disk=22293117907,mem=11964K,pages=19,vmem=356724K|cpu=04:21:01,energy=0,fs/disk=22293117907,mem=4102096K,pages=7711,vmem=37743716K
```

Which gives some information about the resource usage at the end of the job. 

To have a better understanding of the efficiency of resource usage, you can run the following command after the job is finished:

```sh
sacct -j <slurm_job_id> \
  --format=JobID,JobName%30,Partition,AllocCPUS,Elapsed,TotalCPU,MaxRSS,State,ExitCode \
  --parsable2 >> "eso4clima_<slurm_job_id>.out"

```

This will output the resource usage information and add it to the slurm job output file. After running this you can find the line like this in the output file:

```
JobID|JobName|Partition|AllocCPUS|Elapsed|TotalCPU|MaxRSS|State|ExitCode
23743544|eso4clima|compute|256|00:02:44|04:21:01||COMPLETED|0:0
23743544.batch|batch||256|00:02:44|04:21:01|4102096K|COMPLETED|0:0
23743544.extern|extern||256|00:02:44|00:00.001|3752K|COMPLETED|0:0
```

The the efficiency of resource usage can be calculated as `TotalCPU / AllocCPUS * Elapsed Time`. In the example above, the CPU time is `04:21:01`, the allocated CPU is `256`, and the elapsed time is `00:02:44`, so the resource usage is `4:21:01 / 256 * 00:02:44 = 0.37`.