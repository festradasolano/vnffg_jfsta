from embed_drl import DRLRun
from embed_drl_jfsta import DRLJointFSTARun
from embed_random import RandomEmbedding
from substrate import build_random_network
from vnfr import build_random_vnfr

from ax.service.ax_client import AxClient, ObjectiveProperties
import os
from pathlib import Path
import sys


# SETTINGS
AX_EXPERIMENT_NAME = "embed_drl_jfsta"
AX_VERBOSE = False
NUM_TRIALS = 30
SEED = 1000
NUM_VNFS = 4

# =============================
# Build random network and VNFR
# =============================
network = build_random_network(seed=SEED)
vnfr = build_random_vnfr(
    nodes_id=network.get_nodes_id(),
    num_vnfs=NUM_VNFS, seed=SEED
)

# Print VNFR information
print()
print("VNFR")
print("\tMin throughput =", vnfr.min_throughput)
print("\tMax delay =", vnfr.max_delay)
print("\tDependencies:", vnfr.dependencies)
print("\tDependants:", vnfr.dependants)

# Check maximum delay
if vnfr.vnfs_delay >= vnfr.max_delay:
    print("\nTERMINATING: VNFs delay is greater than maximum delay; no possibility for improvement.")
    sys.exit(0)

# ===========================
# Build random embedding (M1)
# ===========================

random_embed = RandomEmbedding(list(vnfr.vnfs.keys()), network.vnf_capable_nodes, list(network.links.keys()), seed=SEED)
success = random_embed.build_embbeding(network, vnfr)

if not success:
    print("\nTERMINATING: Unable to build random embedding; something went wrong.")
    sys.exit(-2)

print()
print("Random Embedding")
print("\tThroughput =", random_embed.throughput)
print("\tDelay =", random_embed.delay)
print("\tVNFFG:", random_embed.vnffg)

# Check M1 does not meet QoS requirements
if (random_embed.throughput >= vnfr.min_throughput) and (random_embed.delay <= vnfr.max_delay):
    print("\nTERMINATING: Random embedding meets throughput and delay requirements; no possibility for improvement.")
    sys.exit(0)

# ==============================
# Tune DRL-based JFSTA embedding
# ==============================

# Define hyperparameters
hparams = [
    {
        "name": "hidden_layers",
        "type": "range",
        "value_type": "int",
        "bounds": [2, 5],
    },
    {
        "name": "hidden_neurons_pow2",
        "type": "range",
        "value_type": "int",
        "bounds": [4, 8],  # [16, 32, 64, 128, 256]
    },
    {
        "name": "learning_rate_pow10",
        "type": "range",
        "value_type": "int",
        "bounds": [-5, 0],  # [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1]
    },
    {
        "name": "discount_factor",
        "type": "range",
        "value_type": "float",
        "bounds": [0.9, 0.99],
    },
    {
        "name": "batch_size_pow2",
        "type": "range",
        "value_type": "int",
        "bounds": [3, 7],  # [8, 16, 32, 64, 128]
    },
    {
        "name": "epsilon_decay",
        "type": "range",
        "value_type": "float",
        "bounds": [0.9995, 0.99995],
    },
    {
        "name": "target_net_update_rate_pow2",
        "type": "range",
        "value_type": "int",
        "bounds": [0, 4],  # [50, 100, 200, 400, 800]
    },
]

# Create new Ax experiment
ax_client = AxClient(
    random_seed=SEED,
    verbose_logging=AX_VERBOSE
)
ax_client.create_experiment(
    name=AX_EXPERIMENT_NAME,
    parameters=hparams,
    objectives={
        DRLRun.MAX_AVG_REWARD_KEY: ObjectiveProperties(minimize=False),
        DRLRun.MAX_SUCCESS_RATE_KEY: ObjectiveProperties(minimize=False),
    }
)

# Perform trials to optimize hyperparameters
trial_index = -1
while True:
    # End experiment if we've reached max trials; otherwise, get next hyperparameters
    if trial_index + 1 >= NUM_TRIALS:
        break
    next_hparams, trial_index = ax_client.get_next_trial()

    # Show that we're starting a new trial
    print(f"--- Trial {trial_index} ---")

    # Perform trial
    jfsta = DRLJointFSTARun(
        m1=random_embed,
        network=network,
        vnfr=vnfr,
        seed=SEED
    )
    results = jfsta.run_tuning(next_hparams)
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data=results,
    )

# Export all existing trials
df = ax_client.get_trials_data_frame()
path_results_dir = "{0}/{1}".format(str(Path.home()), DRLRun.RESULTS_DIR_NAME)
if not os.path.exists(path_results_dir):
    os.makedirs(path_results_dir)
filename = "hparamstuning_{0}vnfs_seed{1}.csv".format(NUM_VNFS, SEED)
path_file = "{0}/{1}/{2}".format(str(Path.home()), DRLRun.RESULTS_DIR_NAME, filename)
df.to_csv(path_file)
