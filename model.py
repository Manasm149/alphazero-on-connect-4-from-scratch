"""
AlphaZero on Connect-4 from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_empty_board
import numpy as np

def make_empty_board():
    return np.zeros((6, 7), dtype=int)

# Step 2 - column_top_row
def column_top_row(board, column):
    for row in range(5, -1, -1):
        if board[row, column] == 0:
            return row

    return -1

# Step 3 - drop_piece
def drop_piece(board, column, player):
    row = column_top_row(board, column)

    if row == -1:
        raise ValueError("Column is full")

    new_board = board.copy()
    new_board[row, column] = player

    return new_board

# Step 4 - column_full
def column_full(board, column):
    return column_top_row(board, column) == -1

# Step 5 - valid_moves
def valid_moves(board):
    return [column for column in range(7)
            if not column_full(board, column)]

# Step 6 - four_in_a_row_horizontal
def four_in_a_row_horizontal(board):
    for row in range(6):
        for col in range(4):
            if (board[row, col] != 0 and
                board[row, col] == board[row, col + 1] and
                board[row, col] == board[row, col + 2] and
                board[row, col] == board[row, col + 3]):
                return int(board[row, col])

    return 0

# Step 7 - four_in_a_row_vertical
def four_in_a_row_vertical(board):
    for col in range(7):
        for row in range(3):
            if (board[row, col] != 0 and
                board[row, col] == board[row + 1, col] and
                board[row, col] == board[row + 2, col] and
                board[row, col] == board[row + 3, col]):
                return int(board[row, col])

    return 0

# Step 8 - four_in_a_row_diagonal_down_right
def four_in_a_row_diagonal_down_right(board):
    for row in range(3):
        for col in range(4):
            if (board[row, col] != 0 and
                board[row, col] == board[row + 1, col + 1] and
                board[row, col] == board[row + 2, col + 2] and
                board[row, col] == board[row + 3, col + 3]):
                
                return int(board[row, col])

    return 0

# Step 9 - four_in_a_row_diagonal_up_right
def four_in_a_row_diagonal_up_right(board):
    for row in range(3, 6):
        for col in range(4):
            if (board[row, col] != 0 and
                board[row, col] == board[row - 1, col + 1] and
                board[row, col] == board[row - 2, col + 2] and
                board[row, col] == board[row - 3, col + 3]):

                return int(board[row, col])

    return 0

# Step 10 - check_winner
def check_winner(board):
    winner = four_in_a_row_horizontal(board)
    if winner != 0:
        return winner

    winner = four_in_a_row_vertical(board)
    if winner != 0:
        return winner

    winner = four_in_a_row_diagonal_down_right(board)
    if winner != 0:
        return winner

    winner = four_in_a_row_diagonal_up_right(board)
    if winner != 0:
        return winner

    return 0

# Step 11 - board_is_full
def column_full(board, column):
    return board[0, column] != 0


def valid_moves(board):
    return [column for column in range(7)
            if not column_full(board, column)]


def board_is_full(board):
    return len(valid_moves(board)) == 0

# Step 12 - is_terminal
def is_terminal(board):
    winner = check_winner(board)

    if winner != 0:
        return (True, winner)

    if board_is_full(board):
        return (True, 0)

    return (False, 0)

# Step 13 - other_player
def other_player(player):
    if player == 1:
        return 2
    return 1

# Step 14 - step_env
def step_env(board, column, player):
    # Drop the piece; drop_piece returns a new board
    new_board = drop_piece(board, column, player)

    # Check whether the game is over
    done, winner = is_terminal(new_board)

    # The next player is always the opponent
    next_player = other_player(player)

    return new_board, done, winner, next_player

# Step 15 - encode_board
import numpy as np

def encode_board(board, current_player):
    opponent = other_player(current_player)

    encoded = np.zeros((2, 6, 7), dtype=np.float32)

    encoded[0] = (board == current_player)
    encoded[1] = (board == opponent)

    return encoded

# Step 16 - board_to_torch_tensor
import torch

def board_to_torch_tensor(board, current_player):
    encoded = encode_board(board, current_player)

    tensor = torch.tensor(encoded, dtype=torch.float32)

    return tensor.unsqueeze(0)

# Step 17 - init_conv_backbone
import torch.nn as nn

def init_conv_backbone(in_channels=2, hidden_channels=16):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

# Step 18 - init_policy_head
import torch
import torch.nn as nn


def init_policy_head(hidden_channels=16, num_columns=7):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, num_columns) logits."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, num_columns)),
        nn.Conv2d(hidden_channels, 1, kernel_size=1),
        nn.Flatten(start_dim=1)
    )

# Step 19 - init_value_head
import torch
import torch.nn as nn


def init_value_head(hidden_channels=16):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, 1) in (-1, 1)."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1),
        nn.Linear(hidden_channels, 1),
        nn.Tanh()
    )

# Step 20 - build_policy_value_net
import torch
import torch.nn as nn


def build_policy_value_net(in_channels=2, hidden_channels=16, num_columns=7):
    """Compose backbone + policy head + value head into one nn.Module."""

    class PolicyValueNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.backbone = init_conv_backbone(
                in_channels,
                hidden_channels
            )

            self.policy_head = init_policy_head(
                hidden_channels,
                num_columns
            )

            self.value_head = init_value_head(
                hidden_channels
            )

        def forward(self, x):
            features = self.backbone(x)

            policy_logits = self.policy_head(features)
            value = self.value_head(features)

            return policy_logits, value

    return PolicyValueNet()

# Step 21 - policy_value_forward
import torch
import torch.nn as nn


def policy_value_forward(net, encoded_board):
    """Run encoded_board (B,2,6,7) through net and return (logits, value)."""
    logits, value = net(encoded_board)

    return logits, value

# Step 22 - action_mask
import numpy as np

def action_mask(board):
    mask = np.zeros(7, dtype=bool)

    for column in valid_moves(board):
        mask[column] = True

    return mask

# Step 23 - masked_policy_logits
import torch
import numpy as np

def masked_policy_logits(logits, mask):
    mask = torch.as_tensor(
        mask,
        dtype=torch.bool,
        device=logits.device
    )

    masked_logits = logits.clone()
    masked_logits[..., ~mask] = float('-inf')

    return masked_logits

# Step 24 - masked_log_softmax
import torch

def masked_log_softmax(logits, mask):
    masked_logits = masked_policy_logits(logits, mask)

    return torch.log_softmax(masked_logits, dim=-1)

# Step 25 - sample_action_from_policy
import torch


def sample_action_from_policy(logits, mask, temperature=1.0):
    """Sample a legal column from a tempered masked categorical policy."""

    # Apply temperature
    scaled_logits = logits / temperature

    # Mask illegal columns
    masked_logits = masked_policy_logits(scaled_logits, mask)

    # Convert logits to probabilities
    probabilities = torch.softmax(masked_logits, dim=-1)

    # Sample one action
    action = torch.multinomial(probabilities, num_samples=1)

    return action.item()

# Step 26 - greedy_action_from_policy
import torch

def greedy_action_from_policy(logits, mask):
    masked_logits = masked_policy_logits(logits, mask)
    return int(torch.argmax(masked_logits, dim=-1).item())
def sample_action_from_policy(logits, mask, temperature=1.0):
    """Sample a legal column from a tempered masked categorical policy."""

    # Apply temperature
    scaled_logits = logits / temperature

    # Mask illegal columns
    masked_logits = masked_policy_logits(scaled_logits, mask)

    # Convert logits to probabilities
    probabilities = torch.softmax(masked_logits, dim=-1)

    # Sample one action
    action = torch.multinomial(probabilities, num_samples=1)

    return action.item()

# Step 27 - make_mcts_node
def make_mcts_node(prior=0.0, parent=None):
    return {
        'prior': prior,
        'visit_count': 0,
        'value_sum': 0.0,
        'children': {},
        'parent': parent,
        'is_expanded': False
    }

# Step 28 - node_q_value
def node_q_value(node):
    if node['visit_count'] == 0:
        return 0.0

    return node['value_sum'] / node['visit_count']

# Step 29 - ucb_score
import math

def ucb_score(parent, child, c_puct=1.5):
    q = node_q_value(child)

    exploration = (
        c_puct
        * child['prior']
        * math.sqrt(parent['visit_count'])
        / (1 + child['visit_count'])
    )

    return float(q + exploration)

# Step 30 - select_best_child
def select_best_child(node, legal_actions, c_puct=1.5):
    best_action = None
    best_child = None
    best_score = float('-inf')

    for action in legal_actions:
        child = node['children'][action]

        score = ucb_score(node, child, c_puct)

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child

# Step 31 - select_leaf
def select_leaf(root, c_puct=1.5):
    node = root

    while node.get('is_expanded', False):
        legal_actions = list(node['children'].keys())
        _, node = select_best_child(node, legal_actions, c_puct)

    return node

# Step 32 - evaluate_with_network
import numpy as np
import torch

def evaluate_with_network(net, state, to_play):
    # Encode board and add batch dimension
    encoded = board_to_torch_tensor(state, to_play)

    # Run network
    logits, value = policy_value_forward(net, encoded)

    # Legal move mask
    mask = action_mask(state)

    # Convert logits to log-probabilities, respecting legal moves
    log_probs = masked_log_softmax(logits, mask)

    # Convert log-probabilities to probabilities
    priors = torch.exp(log_probs)

    # Remove batch dimension and convert to numpy
    priors = priors.squeeze(0).detach().cpu().numpy()

    # Extract scalar value
    value = float(value.squeeze().detach().cpu().item())

    return priors, value

# Step 33 - expand_node
def expand_node(node, priors):
    board = node['board']
    player = node['to_play']

    for action in valid_moves(board):
        new_board = drop_piece(board, action, player)
        next_player = other_player(player)

        child = make_mcts_node(
            prior=float(priors[action]),
            parent=node
        )

        child['board'] = new_board
        child['to_play'] = next_player

        node['children'][action] = child

    node['is_expanded'] = True

# Step 34 - backup_value
def backup_value(leaf, value):
    node = leaf

    while node is not None:
        node['visits'] += 1
        node['value_sum'] += value

        value = -value
        node = node['parent']
def make_mcts_node(prior=0.0, parent=None):
    return {
        'prior': prior,
        'visits': 0,
        'value_sum': 0.0,
        'children': {},
        'parent': parent
        
    }

# Step 35 - run_one_simulation
def run_one_simulation(root, net, c_puct=1.5):
    leaf = select_leaf(root, c_puct)

    board = leaf['board']
    to_play = leaf['to_play']

    done, winner = is_terminal(board)

    if done:
        if winner == 0:
            value = 0.0
        elif winner == to_play:
            value = 1.0
        else:
            value = -1.0
    else:
        priors, value = evaluate_with_network(
            net,
            board,
            to_play
        )

        expand_node(leaf, priors)

    backup_value(leaf, value)

# Step 36 - run_mcts
def run_mcts(state, to_play, net, num_simulations, c_puct):
    root = make_mcts_node()

    root['board'] = state.copy()
    root['to_play'] = to_play

    for _ in range(num_simulations):
        run_one_simulation(root, net, c_puct)

    return root

# Step 37 - visit_count_policy
import numpy as np

def visit_count_policy(root, temperature=1.0):
    probs = np.zeros(7, dtype=float)

    children = root['children']

    # No children → uniform policy
    if not children:
        return np.ones(7, dtype=float) / 7.0

    # Temperature = 0 → greedy / argmax
    if temperature == 0:
        best_action = max(
            children,
            key=lambda action: children[action]['visit_count']
        )
        probs[best_action] = 1.0
        return probs

    # Temperature > 0
    for action, child in children.items():
        visits = child['visit_count']
        probs[action] = visits ** (1.0 / temperature)

    total = probs.sum()

    # Safety fallback if all visit counts are zero
    if total == 0:
        return np.ones(7, dtype=float) / 7.0

    return probs / total

# Step 38 - mcts_choose_action
def mcts_choose_action(state, to_play, net, num_simulations, c_puct, temperature=1.0):
    root = run_mcts(
        state,
        to_play,
        net,
        num_simulations,
        c_puct
    )

    policy = visit_count_policy(
        root,
        temperature
    )

    action = int(np.random.choice(7, p=policy))

    return action, policy

# Step 39 - record_self_play_step
def record_self_play_step(history, board, policy, to_play):
    history.append({
        'board': board.copy(),
        'policy': policy.copy(),
        'to_play': to_play
    })

    return history

# Step 40 - play_self_play_game
def play_self_play_game(net, num_simulations, c_puct, temperature=1.0):
    board = make_empty_board()
    to_play = 1
    history = []

    while True:
        # Choose action and get the MCTS policy for the current position
        action, policy = mcts_choose_action(
            board,
            to_play,
            net,
            num_simulations,
            c_puct,
            temperature
        )

        # Record the position BEFORE making the move
        record_self_play_step(
            history,
            board,
            policy,
            to_play
        )

        # Apply the move
        board, done, winner, next_player = step_env(
            board,
            action,
            to_play
        )

        if done:
            return history, winner

        to_play = next_player

# Step 41 - assign_value_targets
def assign_value_targets(history, winner):
    result = []

    for step in history:
        new_step = step.copy()

        if winner == 0:
            value = 0.0
        elif step['to_play'] == winner:
            value = 1.0
        else:
            value = -1.0

        new_step['value'] = value
        result.append(new_step)

    return result

# Step 42 - generate_self_play_batch
def generate_self_play_batch(net, num_games, num_simulations, c_puct, temperature=1.0):
    buffer = []

    for _ in range(num_games):
        history, winner = play_self_play_game(
            net,
            num_simulations,
            c_puct,
            temperature
        )

        labelled_history = assign_value_targets(history, winner)
        buffer.extend(labelled_history)

    return buffer

# Step 43 - value_loss_mse
import torch

def value_loss_mse(predicted_values, target_values):
    return torch.mean((predicted_values - target_values) ** 2)

# Step 44 - policy_loss_cross_entropy
import torch

def policy_loss_cross_entropy(predicted_log_probs, target_policy):
    return -(target_policy * predicted_log_probs).sum(dim=1).mean()

# Step 45 - l2_regularization_loss
import torch

def l2_regularization_loss(net):
    loss = torch.tensor(0.0, device=next(net.parameters()).device)

    for param in net.parameters():
        if param.requires_grad:
            loss = loss + torch.sum(param ** 2)

    return loss

# Step 46 - combined_loss
def combined_loss(
    predicted_log_probs,
    predicted_values,
    target_policy,
    target_values,
    net,
    policy_weight=1.0,
    value_weight=1.0,
    l2_weight=1e-4
):
    policy_loss = policy_loss_cross_entropy(
        predicted_log_probs,
        target_policy
    )

    value_loss = value_loss_mse(
        predicted_values,
        target_values
    )

    l2_loss = l2_regularization_loss(net)

    total_loss = (
        policy_weight * policy_loss
        + value_weight * value_loss
        + l2_weight * l2_loss
    )

    components = {
        'policy': policy_loss,
        'value': value_loss,
        'l2': l2_loss
    }

    return total_loss, components

# Step 47 - encode_batch_states
import numpy as np
import torch

def encode_batch_states(boards, to_plays):
    encoded = [
        encode_board(board, player)
        for board, player in zip(boards, to_plays)
    ]

    return torch.tensor(
        np.stack(encoded),
        dtype=torch.float32
    )

# Step 48 - iterate_minibatches
import numpy as np

def iterate_minibatches(buffer, batch_size, seed=None):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(buffer))
    rng.shuffle(indices)

    for start in range(0, len(buffer), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield [buffer[i] for i in batch_indices]

# Step 49 - training_step
import numpy as np
import torch


def training_step(
    net,
    optimizer,
    minibatch,
    policy_weight=1.0,
    value_weight=1.0,
    l2_weight=1e-4
):
    boards = [step['board'] for step in minibatch]
    to_plays = [step['to_play'] for step in minibatch]

    target_policy = torch.tensor(
        np.stack([step['policy'] for step in minibatch]),
        dtype=torch.float32
    )

    target_values = torch.tensor(
        [step['value'] for step in minibatch],
        dtype=torch.float32
    )

    # Encode boards: (B, 2, 6, 7)
    states = encode_batch_states(boards, to_plays)

    # Forward pass
    logits, predicted_values = policy_value_forward(net, states)

    # Build one legal-move mask per board
    masks = np.stack([
        action_mask(board)
        for board in boards
    ])

    # Mask illegal columns before log-softmax
    masked_logits = masked_policy_logits(logits, masks)
    predicted_log_probs = torch.log_softmax(masked_logits, dim=-1)

    predicted_values = predicted_values.squeeze(-1)

    # Individual loss components
    policy_loss = policy_loss_cross_entropy(
        predicted_log_probs,
        target_policy
    )

    value_loss = value_loss_mse(
        predicted_values,
        target_values
    )

    l2_loss = l2_regularization_loss(net)

    # Combined loss
    total_loss = (
        policy_weight * policy_loss
        + value_weight * value_loss
        + l2_weight * l2_loss
    )

    # One optimizer update
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        'total': float(total_loss.detach().item()),
        'policy': float(policy_loss.detach().item()),
        'value': float(value_loss.detach().item()),
        'l2': float(l2_loss.detach().item())
    }

# Step 50 - training_epoch
def training_epoch(
    net,
    optimizer,
    buffer,
    batch_size,
    seed=None,
    policy_weight=1.0,
    value_weight=1.0,
    l2_weight=1e-4
):
    totals = {
        'total': 0.0,
        'policy': 0.0,
        'value': 0.0,
        'l2': 0.0
    }

    num_batches = 0

    for minibatch in iterate_minibatches(buffer, batch_size, seed):
        losses = training_step(
            net,
            optimizer,
            minibatch,
            policy_weight,
            value_weight,
            l2_weight
        )

        for key in totals:
            totals[key] += losses[key]

        num_batches += 1

    # Handle an empty buffer
    if num_batches == 0:
        return totals

    for key in totals:
        totals[key] /= num_batches

    return totals

# Step 51 - self_play_iteration
def self_play_iteration(
    net,
    optimizer,
    num_games,
    num_simulations,
    c_puct,
    batch_size,
    num_epochs=1,
    temperature=1.0
):
    buffer = generate_self_play_batch(
        net,
        num_games,
        num_simulations,
        c_puct,
        temperature
    )

    losses = []

    for _ in range(num_epochs):
        epoch_losses = training_epoch(
            net,
            optimizer,
            buffer,
            batch_size
        )
        losses.append(epoch_losses)

    return {
        'buffer_size': len(buffer),
        'losses': losses
    }

# Step 52 - train_loop
def train_loop(
    net,
    optimizer,
    num_iterations,
    num_games,
    num_simulations,
    c_puct,
    batch_size,
    num_epochs=1,
    temperature=1.0
):
    results = []

    for _ in range(num_iterations):
        result = self_play_iteration(
            net,
            optimizer,
            num_games,
            num_simulations,
            c_puct,
            batch_size,
            num_epochs,
            temperature
        )

        results.append(result)

    return results

# Step 53 - random_policy_action
import numpy as np

def random_policy_action(state, to_play, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    legal_moves = valid_moves(state)
    return int(rng.choice(legal_moves))

# Step 54 - greedy_agent_action
def greedy_agent_action(net, state, to_play):
    encoded = encode_board(state, to_play)

    # Add batch dimension: (1, 2, 6, 7)
    encoded = torch.as_tensor(
        encoded, dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        logits, _ = policy_value_forward(net, encoded)

    legal = valid_moves(state)

    # Only consider legal columns
    legal_logits = logits[0, legal]

    return int(legal[torch.argmax(legal_logits).item()])

# Step 55 - play_one_match (not yet solved)
# TODO: implement

# Step 56 - match_win_rate (not yet solved)
# TODO: implement

# Step 57 - evaluate_against_random (not yet solved)
# TODO: implement

