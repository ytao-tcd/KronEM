import random
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.nn import init
import numpy as np
import copy
import math
import sys
import networkx as nx

class KroneckerGenerator(nn.Module):
    """
    A PyTorch module to generate a large graph probability matrix
    by taking the Kronecker power of a small initiator probability matrix.
    """
    def __init__(self, p0, k_order=3):
        super(KroneckerGenerator, self).__init__()
        self.p = Parameter(p0, requires_grad=True)
        self.k_order = k_order

    def generate_adjacency_probability(self):
        """
        Generates the final probability matrix via Kronecker product.
        """
        p0 = self.p
        adj_prob = self.p
        for _ in range(self.k_order - 1):
            adj_prob = kronecker(adj_prob, p0)
        return adj_prob


def create_missing_mask(num_nodes, missing_percent):
    """
    Creates masks for observed and unobserved portions of the adjacency matrix.
    A node is marked as "missing" if its row and column are entirely unobserved.

    Args:
        num_nodes (int): The number of nodes in the graph.
        missing_percent (float): The fraction of nodes to be marked as unobserved.

    Returns:
        tuple: (mask_unobserved, mask_observed)
    """
    missing_num = int(num_nodes * missing_percent)
    idx = torch.randperm(num_nodes)[:missing_num]
    mask_unobserved = torch.zeros(num_nodes, num_nodes)
    for i in idx:
        mask_unobserved[i, :] = 1  # Mark the i-th row as unobserved
        mask_unobserved[:, i] = 1  # Mark the i-th column as unobserved
    mask_observed = 1 - mask_unobserved
    return mask_unobserved, mask_observed


def generate_kronecker_adj(k_order, initiator_matrix):
    """
    Helper function to generate the full adjacency probability matrix from an initiator matrix.
    """
    p0 = adj = initiator_matrix
    for _ in range(k_order - 1):
        adj = kronecker(adj, p0)
    return adj


def kronecker(A, B):
    """
    Computes the Kronecker product of two matrices using torch.einsum for efficiency.
    """
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0) * B.size(0), A.size(1) * B.size(1))


def swap_node_in_matrix(matrix, i, j):
    """
    Swaps node i and node j in a given adjacency matrix.
    This involves swapping rows i and j, and then columns i and j.
    """
    swapped_matrix = copy.deepcopy(matrix)
    # Swap rows
    i_topology = swapped_matrix[i, :].clone()
    j_topology = swapped_matrix[j, :].clone()
    swapped_matrix[i, :] = j_topology
    swapped_matrix[j, :] = i_topology
    # Swap columns
    i_col_topology = swapped_matrix[:, i].clone()
    j_col_topology = swapped_matrix[:, j].clone()
    swapped_matrix[:, i] = j_col_topology
    swapped_matrix[:, j] = i_col_topology
    return swapped_matrix


def get_apx_empty_graph_ll(prob_matrix, k_order):
    """
    Calculates the approximate log-likelihood of an empty graph.
    """
    prob_flat = prob_matrix.flatten()
    sum_p = float(prob_flat.sum())
    sum_sq_p = float((prob_flat**2).sum())
    return -sum_p**k_order - 0.5 * sum_sq_p**k_order


def get_apx_no_edge_ll(node_id1, node_id2, k_order, ll_matrix):
    """
    Calculates the approximate log-likelihood of a non-edge.
    """
    edge_ll = get_edge_ll(node_id1, node_id2, k_order, ll_matrix)
    return -math.exp(edge_ll) - 0.5 * math.exp(2 * edge_ll)


def get_edge_ll(node_id1, node_id2, k_order, ll_matrix):
    """
    Calculates the exact log-likelihood of a single edge (i, j).
    This is derived from the initiator's log-likelihood matrix by mapping
    the binary representations of node_id1 and node_id2.
    """
    ll = 0
    # Get binary representations of node IDs
    node_id1_bin = bin(node_id1)[2:].zfill(k_order)
    node_id2_bin = bin(node_id2)[2:].zfill(k_order)
    for i in range(k_order):
        ll += ll_matrix[int(node_id1_bin[i]), int(node_id2_bin[i])]
    return ll


def calc_apx_graph_ll(prob_matrix, k_order, adj_matrix, p0_ll_matrix):
    """
    Calculates the approximate log-likelihood of the entire graph.
    """
    prob_mtx_dim = len(prob_matrix)
    log_like = get_apx_empty_graph_ll(prob_matrix, k_order)
    edge_set = torch.nonzero(adj_matrix)
    for edge in edge_set:
        log_like -= get_apx_no_edge_ll(edge[0], edge[1], k_order, p0_ll_matrix)
        log_like += get_edge_ll(edge[0], edge[1], k_order, p0_ll_matrix)
    return log_like


def node_ll_delta(node_id, adj_matrix, ll_matrix, k_order):
    """
    Calculates the change in log-likelihood associated with a single node.
    """
    if adj_matrix[:, node_id].sum() == 0 and adj_matrix[node_id, :].sum() == 0:
        return 0.0
    delta = 0
    num_nodes = len(adj_matrix)
    for i in range(num_nodes):
        if adj_matrix[node_id][i] == 1:
            delta += -get_apx_no_edge_ll(node_id, i, k_order, ll_matrix) + get_edge_ll(node_id, i, k_order, ll_matrix)
    for i in range(num_nodes):
        if adj_matrix[i][node_id] == 1:
            delta += -get_apx_no_edge_ll(i, node_id, k_order, ll_matrix) + get_edge_ll(i, node_id, k_order, ll_matrix)
    if adj_matrix[node_id][node_id] == 1:
        # Correct for double counting the self-loop
        delta += get_apx_no_edge_ll(node_id, node_id, k_order, ll_matrix) - get_edge_ll(node_id, node_id, k_order, ll_matrix)
    return delta


def swap_nodes_ll(node_id1, node_id2, log_like, adj_matrix, ll_matrix, k_order):
    """
    Calculates the new log-likelihood after swapping two nodes.
    It efficiently updates the LL by subtracting the contributions of the
    old node positions and adding the contributions of the new positions.
    """
    # Subtract contribution of nodes at old positions
    log_like -= (node_ll_delta(node_id1, adj_matrix, ll_matrix, k_order) +
                 node_ll_delta(node_id2, adj_matrix, ll_matrix, k_order))
    if adj_matrix[node_id1][node_id2] == 1:
        log_like += -get_apx_no_edge_ll(node_id1, node_id2, k_order, ll_matrix) + get_edge_ll(node_id1, node_id2, k_order, ll_matrix)
    if adj_matrix[node_id2][node_id1] == 1:
        log_like += -get_apx_no_edge_ll(node_id2, node_id1, k_order, ll_matrix) + get_edge_ll(node_id2, node_id1, k_order, ll_matrix)

    # Perform the swap
    adj_matrix_swapped = swap_node_in_matrix(adj_matrix, node_id1, node_id2)

    # Add contribution of nodes at new positions
    log_like += (node_ll_delta(node_id1, adj_matrix_swapped, ll_matrix, k_order) +
                 node_ll_delta(node_id2, adj_matrix_swapped, ll_matrix, k_order))
    if adj_matrix_swapped[node_id1][node_id2] == 1:
        log_like += get_apx_no_edge_ll(node_id1, node_id2, k_order, ll_matrix) - get_edge_ll(node_id1, node_id2, k_order, ll_matrix)
    if adj_matrix_swapped[node_id2][node_id1] == 1:
        log_like += get_apx_no_edge_ll(node_id2, node_id1, k_order, ll_matrix) - get_edge_ll(node_id2, node_id1, k_order, ll_matrix)

    return log_like, adj_matrix_swapped


def sample_next_permutation(num_nodes, log_like, adj_matrix, p0_ll_matrix, k_order):
    """
    Performs one step of Metropolis-Hastings sampling for node permutations.
    A random pair of nodes is chosen to be swapped, and the swap is
    accepted or rejected based on the change in graph log-likelihood.
    """
    adj_matrix_old = copy.deepcopy(adj_matrix)
    node_id1, node_id2 = np.random.choice(num_nodes, 2, replace=False)

    old_ll = log_like
    new_ll, adj_matrix_new = swap_nodes_ll(node_id1, node_id2, log_like, adj_matrix, p0_ll_matrix, k_order)

    # Metropolis-Hastings acceptance criteria
    if math.log(random.uniform(0, 1)) > new_ll - old_ll:
        # Reject the swap
        return old_ll, adj_matrix_old, False, node_id1, node_id2
    else:
        # Accept the swap
        return new_ll, adj_matrix_new, True, node_id1, node_id2


def initialize_missing_edges(observed_adj, unobserved_mask, all_unobserved_adj):
    """
    Initializes the unobserved part of the graph (Z) with a random
    subset of edges, constrained to have the same total number of edges
    as the ground truth in that region.
    """
    num_missing_edges = int(all_unobserved_adj.sum())
    unobserved_indices = torch.nonzero(unobserved_mask)
    num_possible_edges = len(unobserved_indices)

    # Randomly select indices to place edges
    choice_indices = torch.randperm(num_possible_edges)[:num_missing_edges]
    initial_edge_indices = torch.index_select(unobserved_indices, 0, choice_indices)

    # Start with the observed graph and add the initialized missing edges
    H = copy.deepcopy(observed_adj)
    H[initial_edge_indices[:, 0], initial_edge_indices[:, 1]] = 1
    return H


def metro_gibbs_setup(warmup_iters, k_order, observed_adj, unobserved_mask, all_unobserved_adj, log_file_handle):
    """
    Initializes the E-step by setting up the state for MCMC sampling.
    This includes initializing the missing edges and running a warmup
    phase for the node permutation sampling.
    """
    # Initialize initiator matrix
    prob_matrix = torch.rand([2, 2])
    ll_matrix = torch.log(prob_matrix)
    print(f'Original Initializer Prob Matrix: {prob_matrix}')
    log_file_handle.write(f'Original Initializer Prob Matrix: {prob_matrix}\n')

    H = initialize_missing_edges(observed_adj, unobserved_mask, all_unobserved_adj)
    log_like = calc_apx_graph_ll(prob_matrix, k_order, H, ll_matrix)

    # MCMC warmup phase
    for i in range(warmup_iters):
        log_like, H, _, _, _ = sample_next_permutation(len(observed_adj), log_like, H, ll_matrix, k_order)
        if i % 30 == 0:
            message = f'E-Step: MCMC setup warmup {i} of {warmup_iters}'
            print(message)
            log_file_handle.write(message + '\n')

    return H, ll_matrix, log_like, prob_matrix


def get_apx_no_edge_dll(param_id, node_id1, node_id2, k_order, ll_matrix):
    """
    Calculates the derivative of the approximate log-likelihood of a non-edge
    with respect to a single parameter in the initiator matrix.
    """
    initiator_dim = len(ll_matrix)
    theta_x, theta_y = param_id % initiator_dim, param_id // initiator_dim
    theta_count, dll = 0, 0
    
    n1_copy, n2_copy = node_id1, node_id2
    for _ in range(k_order):
        x, y = n1_copy % initiator_dim, n2_copy % initiator_dim
        l_val = ll_matrix[x][y]
        if x == theta_x and y == theta_y:
            if theta_count != 0:
                dll += l_val
            theta_count += 1
        else:
            dll += l_val
        n1_copy //= initiator_dim
        n2_copy //= initiator_dim
        
    term1 = -theta_count * torch.exp(dll)
    term2 = -theta_count * torch.exp(ll_matrix[theta_x][theta_y] + 2 * dll)
    return term1 + term2


def get_edge_dll(param_id, node_id1, node_id2, k_order, ll_matrix):
    """
    Calculates the derivative of the log-likelihood of an edge
    with respect to a single parameter in the initiator matrix.
    """
    initiator_dim = len(ll_matrix)
    theta_x, theta_y = param_id % initiator_dim, param_id // initiator_dim
    theta_count = 0.0

    n1_copy, n2_copy = node_id1, node_id2
    for _ in range(k_order):
        if (n1_copy % initiator_dim) == theta_x and (n2_copy % initiator_dim) == theta_y:
            theta_count += 1
        n1_copy //= initiator_dim
        n2_copy //= initiator_dim

    # The derivative of log(p) w.r.t p is 1/p.
    return theta_count / torch.exp(ll_matrix.flatten()[param_id])


def node_dll_delta(param_id, node_id, adj_matrix, k_order, ll_matrix):
    """
    Calculates the change in the derivative of the log-likelihood
    associated with a single node.
    """
    if adj_matrix[:, node_id].sum() == 0 and adj_matrix[node_id, :].sum() == 0:
        return 0.0
    
    delta = 0.0
    num_nodes = len(adj_matrix)
    for i in range(num_nodes):
        if adj_matrix[node_id, i] == 1:
            delta += -get_apx_no_edge_dll(param_id, node_id, i, k_order, ll_matrix) + get_edge_dll(param_id, node_id, i, k_order, ll_matrix)
    for i in range(num_nodes):
        if adj_matrix[i, node_id] == 1:
            delta += -get_apx_no_edge_dll(param_id, i, node_id, k_order, ll_matrix) + get_edge_dll(param_id, i, node_id, k_order, ll_matrix)
    if adj_matrix[node_id, node_id]:
        # Correct for double counting
        delta += get_apx_no_edge_dll(param_id, node_id, node_id, k_order, ll_matrix) - get_edge_dll(param_id, node_id, node_id, k_order, ll_matrix)
        
    return delta


def update_graph_dll(swap_nid1, swap_nid2, ll_matrix, grad_vector, adj_matrix, k_order):
    """
    Updates the gradient vector after a node swap.
    """
    num_params = len(ll_matrix.flatten())
    updated_grad = grad_vector.clone()
    
    for param_id in range(num_params):
        dll = updated_grad[param_id]
        
        # Create the swapped matrix once for this param_id
        swapped_adj = swap_node_in_matrix(adj_matrix, swap_nid1, swap_nid2)

        # Subtract old contributions
        dll -= (node_dll_delta(param_id, swap_nid1, adj_matrix, k_order, ll_matrix) +
                node_dll_delta(param_id, swap_nid2, adj_matrix, k_order, ll_matrix))
        if adj_matrix[swap_nid1, swap_nid2] == 1:
            dll += -get_apx_no_edge_dll(param_id, swap_nid1, swap_nid2, k_order, ll_matrix) + get_edge_dll(param_id, swap_nid1, swap_nid2, k_order, ll_matrix)
        if adj_matrix[swap_nid2, swap_nid1] == 1:
            dll += -get_apx_no_edge_dll(param_id, swap_nid2, swap_nid1, k_order, ll_matrix) + get_edge_dll(param_id, swap_nid2, swap_nid1, k_order, ll_matrix)

        # Add new contributions
        dll += (node_dll_delta(param_id, swap_nid1, swapped_adj, k_order, ll_matrix) +
                node_dll_delta(param_id, swap_nid2, swapped_adj, k_order, ll_matrix))
        if swapped_adj[swap_nid1, swap_nid2] == 1:
            dll += get_apx_no_edge_dll(param_id, swap_nid1, swap_nid2, k_order, ll_matrix) - get_edge_dll(param_id, swap_nid1, swap_nid2, k_order, ll_matrix)
        if swapped_adj[swap_nid2, swap_nid1] == 1:
            dll += get_apx_no_edge_dll(param_id, swap_nid2, swap_nid1, k_order, ll_matrix) - get_edge_dll(param_id, swap_nid2, swap_nid1, k_order, ll_matrix)
            
        updated_grad[param_id] = dll
        
    return updated_grad

def metro_gibbs_sample_step(iterations, update_dll, H, prob_matrix, ll_matrix, unobserved_mask, k_order, log_like, grad_vector):
    """
    Performs one full sampling step, which includes both Gibbs sampling for
    edges in the unobserved region and Metropolis-Hastings for node permutations.
    """
    # Part 1: Gibbs sampling for edges
    for _ in range(iterations):
        # Propose removing an edge from the unobserved part
        edges_in_unobserved = (H > 0) * unobserved_mask
        edge_positions = torch.where(edges_in_unobserved)
        if len(edge_positions[0]) == 0: continue # No edges to remove
        
        removed_edge_idx = torch.randint(0, len(edge_positions[0]), (1,)).item()
        removed_edge_pos = (edge_positions[0][removed_edge_idx], edge_positions[1][removed_edge_idx])

        # Propose adding a non-edge, sampled according to the model probability
        # This sampling process is intricate, based on the Kronecker structure
        num_nodes = len(prob_matrix) ** k_order
        initiator_dim = len(prob_matrix)
        
        added_row, added_col = 0, 0
        rng_val = num_nodes
        
        flat_probs = prob_matrix.flatten()
        prob_cumsum = torch.cumsum(flat_probs, dim=0) / flat_probs.sum()
        
        for _ in range(k_order):
            n = 0
            while random.uniform(0, 1) > prob_cumsum[n]:
                n += 1
            rng_val //= initiator_dim
            row_bit = n // initiator_dim
            col_bit = n % initiator_dim
            added_row += row_bit * rng_val
            added_col += col_bit * rng_val
            
        H[added_row, added_col] = 1

        # Accept/reject the edge swap
        ll_removed = get_edge_ll(removed_edge_pos[0], removed_edge_pos[1], k_order, ll_matrix)
        ll_added = get_edge_ll(added_row, added_col, k_order, ll_matrix)
        
        # --- OPTIMIZED LINE ---
        acceptance_prob = (1.0 - math.exp(ll_removed)) / (1.0 - math.exp(ll_added))
        acceptance_prob = min(acceptance_prob, 1.0)
        
        if random.uniform(0, 1) > acceptance_prob:
            # Reject: undo the add
            H[added_row, added_col] = 0
        else:
            # Accept: complete the removal and update LL and gradients
            H[removed_edge_pos[0], removed_edge_pos[1]] = 0
            log_like += get_apx_no_edge_ll(removed_edge_pos[0], removed_edge_pos[1], k_order, ll_matrix) - ll_removed
            log_like += -get_apx_no_edge_ll(added_row, added_col, k_order, ll_matrix) + ll_added
            
            if update_dll:
                for p_id in range(len(ll_matrix.flatten())):
                    grad_vector[p_id] += get_apx_no_edge_dll(p_id, removed_edge_pos[0], removed_edge_pos[1], k_order, ll_matrix) - get_edge_dll(p_id, removed_edge_pos[0], removed_edge_pos[1], k_order, ll_matrix)
                    grad_vector[p_id] += -get_apx_no_edge_dll(p_id, added_row, added_col, k_order, ll_matrix) + get_edge_dll(p_id, added_row, added_col, k_order, ll_matrix)

    # Part 2: Metropolis-Hastings for permutations
    for _ in range(iterations):
        new_log_like, new_H, accepted, nid1, nid2 = sample_next_permutation(len(H), log_like, H, ll_matrix, k_order)
        if accepted:
            log_like, H = new_log_like, new_H
            if update_dll:
                grad_vector = update_graph_dll(nid1, nid2, ll_matrix, grad_vector, H, k_order)
                
    return H, log_like, grad_vector

def get_apx_empty_graph_dll(param_id, prob_matrix, k_order):
    """
    Calculates the derivative of the empty graph log-likelihood w.r.t. one parameter.
    """
    flat_p = prob_matrix.flatten()
    sum_p = flat_p.sum()
    sum_sq_p = (flat_p**2).sum()
    
    term1 = -k_order * (sum_p ** (k_order - 1))
    term2 = -k_order * (sum_sq_p ** (k_order - 1)) * prob_matrix.flatten()[param_id]
    return term1 + term2


def calc_apx_graph_dll(ll_matrix, adj_matrix, k_order, prob_matrix):
    """
    Calculates the full gradient vector of the graph log-likelihood.
    """
    num_params = len(ll_matrix.flatten())
    grad_vector = torch.zeros(num_params)
    
    for param_id in range(num_params):
        dll = get_apx_empty_graph_dll(param_id, prob_matrix, k_order)
        edge_set = torch.nonzero(adj_matrix)
        for edge in edge_set:
            dll -= get_apx_no_edge_dll(param_id, edge[0], edge[1], k_order, ll_matrix)
            dll += get_edge_dll(param_id, edge[0], edge[1], k_order, ll_matrix)
        grad_vector[param_id] = dll
        
    return grad_vector


def e_step(observed_adj, num_samples, warmup_iters, unobserved_mask, all_unobserved_adj, k_order, log_file_handle, is_first_iter, prev_state=None):
    """
    Performs the Expectation (E) step of the EM algorithm.
    It uses MCMC to sample complete graphs (H) and estimate the
    expected log-likelihood and its gradient.
    """
    if is_first_iter:
        H, ll_matrix, log_like, prob_matrix = metro_gibbs_setup(warmup_iters, k_order, observed_adj, unobserved_mask, all_unobserved_adj, log_file_handle)
    else:
        H, ll_matrix, log_like, prob_matrix = prev_state['H'], prev_state['ll_matrix'], prev_state['log_like'], prev_state['prob_matrix']

    grad_vector = torch.zeros(len(prob_matrix.flatten()))
    
    # MCMC sampling phase
    for i in range(warmup_iters):
        H, log_like, grad_vector = metro_gibbs_sample_step(10, False, H, prob_matrix, ll_matrix, unobserved_mask, k_order, log_like, grad_vector)
        if i % 30 == 0:
            message = f'E-Step: MCMC sampling warmup {i} of {warmup_iters}'
            print(message)
            log_file_handle.write(message + '\n')

    # Re-calculate LL and Gradient after warmup
    log_like = calc_apx_graph_ll(prob_matrix, k_order, H, ll_matrix)
    grad_vector = calc_apx_graph_dll(ll_matrix, H, k_order, prob_matrix)

    ll_samples, dll_samples = [], []
    for i in range(num_samples):
        H, log_like, grad_vector = metro_gibbs_sample_step(50, True, H, prob_matrix, ll_matrix, unobserved_mask, k_order, log_like, grad_vector)
        if i % 30 == 0:
            message = f'E-Step: MCMC sample collection {i} of {num_samples}'
            print(message)
            log_file_handle.write(message + '\n')
        ll_samples.append(log_like)
        dll_samples.append(grad_vector.clone())
        
    current_state = {'H': H, 'll_matrix': ll_matrix, 'log_like': log_like, 'prob_matrix': prob_matrix}
    return current_state, ll_samples, dll_samples


def m_step(prev_state, ll_samples, dll_samples, grad_ascent_iters, learning_rate, min_step, max_step, 
           num_real_edges, num_real_nodes, unobserved_mask, k_order, log_file_handle):
    """
    Performs the Maximization (M) step of the EM algorithm.
    It uses gradient ascent to find the initiator matrix parameters that
    maximize the expected log-likelihood found in the E-step.
    """
    prob_matrix = prev_state['prob_matrix']
    H = prev_state['H']
    num_params = len(prob_matrix.flatten())

    # Average the log-likelihood and gradients over all samples
    avg_ll = sum(ll_samples) / len(ll_samples)
    avg_grad = torch.stack(dll_samples).mean(dim=0)
    
    old_ll = avg_ll
    max_ll = avg_ll
    max_prob_matrix = prob_matrix.clone()
    
    learning_rates = torch.full((num_params,), learning_rate)

    for iter_num in range(grad_ascent_iters):
        message = f"    M-Step Iter {iter_num+1}]"
        print(message)
        log_file_handle.write(message + '\n')

        old_prob_matrix = prob_matrix.clone()
        
        # Update parameters using gradient ascent with adaptive learning rate
        for p in range(num_params):
            step_size = learning_rates[p] * avg_grad[p]
            
            # Adapt learning rate
            while abs(step_size) > max_step:
                learning_rates[p] *= 0.95
                step_size = learning_rates[p] * avg_grad[p]
            
            # Update probability, ensuring it stays within bounds (0, 1)
            p_row, p_col = p // len(prob_matrix), p % len(prob_matrix)
            new_val = prob_matrix[p_row, p_col] + step_size
            prob_matrix[p_row, p_col] = torch.clamp(new_val, 0.0001, 0.9999)

        # Re-evaluate LL and gradient with new parameters (inter-loop update)
        if iter_num < grad_ascent_iters - 1:
            log_like = prev_state['log_like']
            grad_vector = calc_apx_graph_dll(torch.log(prob_matrix), H, k_order, prob_matrix) # Start with base grad
            
            ll_sum, grad_sum = 0, torch.zeros(num_params)
            num_recalc_samples = len(ll_samples) // 10
            
            for s_idx in range(len(ll_samples)):
                H, log_like, grad_vector = metro_gibbs_sample_step(10, True, H, old_prob_matrix, torch.log(old_prob_matrix), unobserved_mask, k_order, log_like, grad_vector)
                
                # Recalculate LL and gradient periodically for accuracy
                if s_idx % num_recalc_samples == 0:
                    current_ll = calc_apx_graph_ll(prob_matrix, k_order, H, torch.log(prob_matrix))
                    current_dll = calc_apx_graph_dll(torch.log(prob_matrix), H, k_order, prob_matrix)
                else: # Use faster approximation otherwise
                    current_ll = log_like 
                    current_dll = grad_vector
                
                ll_sum += current_ll
                grad_sum += current_dll
            
            avg_ll = ll_sum / len(ll_samples)
            avg_grad = grad_sum / len(ll_samples)

        # Logging and tracking best parameters
        if avg_ll > max_ll:
            max_ll = avg_ll
            max_prob_matrix = prob_matrix.clone()
            
        print(f"      Current LL: {avg_ll:.4f}, Delta LL: {avg_ll - old_ll:.4f}")
        log_file_handle.write(f"      Current LL: {avg_ll:.4f}, Delta LL: {avg_ll - old_ll:.4f}\n")
        old_ll = avg_ll

    final_prob_matrix = max_prob_matrix
    final_ll_matrix = torch.log(final_prob_matrix)
    
    print(f'    Final LL: {max_ll:.4f}')
    log_file_handle.write(f'    Final LL: {max_ll:.4f}\n')
    print(f'    FITTED PARAMS: {final_prob_matrix}')
    log_file_handle.write(f'    FITTED PARAMS: {final_prob_matrix}\n')

    # Update state for the next E-step
    updated_state = {
        'H': H, 
        'll_matrix': final_ll_matrix, 
        'log_like': max_ll, 
        'prob_matrix': final_prob_matrix
    }
    return updated_state


def calculate_auc(true_adj, prob_adj, evaluation_mask):
    """
    Calculates the Area Under the ROC Curve (AUC).
    """
    true_labels = true_adj[evaluation_mask].flatten()
    pred_scores = prob_adj[evaluation_mask].flatten()
    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    return auc(fpr, tpr)


def kronEM(em_iterations, num_samples, warmup_iters, observed_adj, unobserved_mask, all_unobserved_adj,
           learning_rate, k_order, grad_ascent_iters, min_step, max_step, num_real_edges, num_real_nodes, 
           ground_truth_adj, log_file_handle):
    """
    The main KronEM algorithm loop.
    """
    prev_state = None
    
    for i in range(em_iterations):
        print('\n' + '-'*70 + '\n')
        em_iter_msg = f"{i + 1} EM-iter] E-Step"
        print(em_iter_msg)
        log_file_handle.write(em_iter_msg + '\n')

        is_first_iter = (i == 0)
        current_state, ll_samples, dll_samples = e_step(
            observed_adj, num_samples, warmup_iters, unobserved_mask, all_unobserved_adj, k_order, log_file_handle, is_first_iter, prev_state
        )

        print("\n\n")
        m_step_msg = f"{i + 1:03d} EM-iter] M-Step"
        print(m_step_msg)
        log_file_handle.write(m_step_msg + '\n')

        updated_state = m_step(
            current_state, ll_samples, dll_samples, grad_ascent_iters, learning_rate,
            min_step, max_step, num_real_edges, num_real_nodes, unobserved_mask, k_order, log_file_handle
        )
        prev_state = updated_state
        
        # Evaluation
        fitted_prob_matrix = updated_state['prob_matrix']
        reconstructed_prob_adj = generate_kronecker_adj(k_order, fitted_prob_matrix)
        
        # AUC on the unobserved (Z) part
        auc_z = calculate_auc(ground_truth_adj, reconstructed_prob_adj, unobserved_mask.bool())
        # AUC on the full graph
        full_mask = torch.ones_like(unobserved_mask).bool()
        auc_full = calculate_auc(ground_truth_adj, reconstructed_prob_adj, full_mask)
        
        result_msg = (f'In KronEM iter {i+1}, Full AUC is {auc_full:.4f}, Unobserved AUC is {auc_z:.4f}\n\n')
        print('-'*45 + '\n' + result_msg)
        log_file_handle.write('-'*45 + '\n' + result_msg)

    return prev_state


if __name__ == "__main__":
    # --- Configuration ---
    INITIATOR_DIM = 2
    K_ORDER = 10
    NUM_NODES = INITIATOR_DIM ** K_ORDER
    EM_ITERATIONS = 30
    REMOVE_PROPORTION = 0.25

    # M-Step parameters
    GRAD_ASCENT_ITERS = 2
    MIN_STEP = 0.001
    MAX_STEP = 0.008
    LEARNING_RATE = 1e-5

    # E-Step parameters
    MCMC_WARMUP_ITERS = 2
    MCMC_NUM_SAMPLES = 20
    
    # --- Ground Truth Graph Generation ---
    # Experiment 1: Barabasi-Albert Graph
    print("Generating Barabasi-Albert graph...")
    graph_nx = nx.random_graphs.barabasi_albert_graph(NUM_NODES, 2)
    adj_matrix_numpy = nx.to_numpy_array(graph_nx)
    ground_truth_adj = torch.from_numpy(adj_matrix_numpy)

    # # Experiment 2: Load from file (example)
    # print("Loading graph from file...")
    # with open('test.txt') as f:
    #     lines = f.readlines()[4:]
    # adj_matrix_numpy = np.zeros((NUM_NODES, NUM_NODES))
    # for line in lines:
    #     src, dst = map(int, line.strip().split('\t'))
    #     adj_matrix_numpy[src, dst] = 1
    # ground_truth_adj = torch.from_numpy(adj_matrix_numpy)

    # # Experiment 3: Watts-Strogatz Graph
    # print("Generating Watts-Strogatz graph...")
    # graph_nx = nx.watts_strogatz_graph(NUM_NODES, 4, 0.2)
    # adj_matrix_numpy = nx.to_numpy_array(graph_nx)
    # ground_truth_adj = torch.from_numpy(adj_matrix_numpy)

    # # Experiment 4: Synthetic Kronecker Graph
    # print("Generating synthetic Kronecker graph...")
    # true_initiator = torch.tensor([[0.9, 0.7], [0.5, 0.2]])
    # ground_adj_prob = generate_kronecker_adj(K_ORDER, true_initiator)
    # ground_truth_adj = (ground_adj_prob > torch.rand_like(ground_adj_prob)).float()
    
    # --- Data Preparation ---
    print("Preparing observed and unobserved masks...")
    unobserved_mask, observed_mask = create_missing_mask(NUM_NODES, REMOVE_PROPORTION)
    
    # G is the observed part of the graph
    observed_adj = ground_truth_adj * observed_mask
    # Z is the unobserved (hidden) part of the graph
    all_unobserved_adj = ground_truth_adj * unobserved_mask

    # --- Run KronEM ---
    log_filename = "kronem_log_BA.txt"
    with open(log_filename, "w") as log_file:
        print(f"Starting KronEM algorithm. Log will be saved to {log_filename}")
        final_state = kronEM(
            em_iterations=EM_ITERATIONS,
            num_samples=MCMC_NUM_SAMPLES,
            warmup_iters=MCMC_WARMUP_ITERS,
            observed_adj=observed_adj,
            unobserved_mask=unobserved_mask.bool(),
            all_unobserved_adj=all_unobserved_adj,
            learning_rate=LEARNING_RATE,
            k_order=K_ORDER,
            grad_ascent_iters=GRAD_ASCENT_ITERS,
            min_step=MIN_STEP,
            max_step=MAX_STEP,
            num_real_edges=observed_adj.sum(),
            num_real_nodes=len(observed_adj) * (1 - REMOVE_PROPORTION),
            ground_truth_adj=ground_truth_adj,
            log_file_handle=log_file
        )
    print("\nKronEM execution finished.")