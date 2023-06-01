import tensorflow as tf
from analyze import analyze_apro
from analyze.utils import get_data


def log_results(
    epsilon,
    architecture,
    ei,
    trained_epsilon,
    test,
    metadata,
    a,
    dataset="mnist",
    batch_size=512,
    augmentation="standard",
    log_file="mnist.log",
):
    tf.config.run_functions_eagerly(True)
    g = analyze_apro(
        dataset=dataset,
        architecture=architecture,
        epsilon=epsilon,
        batch_size=batch_size,
        loss="sparse_trades_ce.1.2",
        augmentation=augmentation,
        load_from=f"/pretrained_model/{dataset}2/{dataset}_{architecture}_e{ei}.h5",
        metadata=metadata,
    )
    eval_result = g.evaluate(test)

    B_list = g.compute_bound()
    print(B_list)

    app_g = analyze_apro(
        dataset=dataset,
        architecture=f'approx_{architecture}.{{"a":{a}, "B_list":{B_list}}}',
        epsilon=epsilon,
        batch_size=batch_size,
        loss="sparse_trades_ce.1.2",
        augmentation=augmentation,
        load_from=f"/pretrained_model/{dataset}2/{dataset}_{architecture.lower()}_e{ei}.h5",
        metadata=metadata,
    )
    app_eval_result = app_g.evaluate(test)

    epsilon = app_g.epsilon
    tolerance = app_g.epsilon * app_g.sub_lipschitz
    total_error = app_g.compute_total_error(0)

    # Write results to log file
    with open(log_file, "a") as f:
        f.write(f"# alpha: {a}\n")
        f.write(
            f"Architecture: {architecture}, Epsilon: {epsilon}, Trained Epsilon: {trained_epsilon}\n"
        )
        f.write(f"g_accuracy: {eval_result[1]:.4f}\n")
        f.write(f"g_vra: {eval_result[2]:.4f}\n")
        f.write(f"app_g_accuracy: {app_eval_result[1]:.4f}\n")
        f.write(f"app_g_vra: {app_eval_result[2]:.4f}\n")
        f.write(f"tolerance: {tolerance:.4f}\n")
        f.write(f"B_list: {B_list}\n")
        f.write(f"total error: {total_error:.4f}\n\n")


# List of epsilons and architectures
epsilons = [
    1 / 255,
    2 / 255,
    3 / 255,
]
trained_epsilons = [
    (0, 1 / 255),
    (1, 2 / 255),
    (2, 3 / 255),
    (3, 4 / 255),
    (4, 5 / 255),
]
a_list = [
    7,
    8,
    9,
    10,
    11,
    12,
]
architectures = ["cnn_2C2F"]

train, test, metadata = get_data("mnist", 512, "standard")

# Loop through each combination of epsilon and architecture and log the results
for architecture in architectures:
    for ei, trained_epsilon in trained_epsilons:
        for a in a_list:
            log_results(
                trained_epsilon, architecture, ei, trained_epsilon, test, metadata, a
            )
