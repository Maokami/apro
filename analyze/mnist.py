import tensorflow as tf
from analyze import analyze_apro
from analyze.utils import get_data


def log_results(
    epsilon,
    architecture,
    ei,
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
        architecture=f'{architecture}.{{"a":{a}}}',
        epsilon=epsilon,
        batch_size=batch_size,
        loss="sparse_trades_ce.1.2",
        augmentation=augmentation,
        load_from=f"/pretrained_model/{dataset}/{dataset}_{architecture}_e{ei}.h5",
        metadata=metadata,
    )
    B_list = g.compute_bound()
    g.set_B(B_list)
    eval_result = g.evaluate(test)

    print(B_list)

    app_g = analyze_apro(
        dataset=dataset,
        architecture=f'approx_{architecture}.{{"a":{a}, "B_list":{B_list}}}',
        epsilon=epsilon,
        batch_size=batch_size,
        loss="sparse_trades_ce.1.2",
        augmentation=augmentation,
        load_from=f"/pretrained_model/{dataset}/{dataset}_{architecture.lower()}_e{ei}.h5",
        metadata=metadata,
    )
    app_eval_result = app_g.evaluate(test)

    epsilon = app_g.epsilon
    tolerance = app_g.epsilon * app_g.sub_lipschitz
    total_error = app_g.compute_total_error(0)

    # Write results to log file
    with open(log_file, "a") as f:
        # f.write(f"# alpha: {a}\n")
        # f.write(f"Architecture: {architecture}, Epsilon: {epsilon}\n")
        # f.write(f"g_accuracy: {eval_result[1]:.4f}\n")
        # f.write(f"g_vra: {eval_result[2]:.4f}\n")
        # f.write(f"app_g_accuracy: {app_eval_result[1]:.4f}\n")
        # f.write(f"app_g_vra: {app_eval_result[2]:.4f}\n")
        # f.write(f"tolerance: {tolerance:.4f}\n")
        # f.write(f"B_list: {B_list}\n")
        # f.write(f"total error: {total_error:.4f}\n\n")
        f.write(
            f"{a}	{B_list}	{architecture}	{tolerance:.4f}	{total_error}	{eval_result[1]:.4f}	{eval_result[2]:.4f}	{app_eval_result[1]:.4f}	{app_eval_result[2]:.4f}\n"
        )


# List of epsilons and architectures
epsilons = [
    (0, 1 / 255),
    (1, 2 / 255),
    (2, 3 / 255),
    (3, 4 / 255),
    (4, 5 / 255),
]
alphas = [
    7,
    8,
    9,
    10,
    11,
    12,
]
architectures = ["cnn_2C2F"]

train, test, metadata = get_data("mnist", 512, "standard")
log_file = "mnist.log"

# Loop through each combination of epsilon and architecture and log the results
for architecture in architectures:
    for ei, epsilon in epsilons:
        for a in alphas:
            log_results(epsilon, architecture, ei, test, metadata, a, log_file=log_file)
