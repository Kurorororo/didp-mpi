use std::collections::BinaryHeap;
use std::fmt::Display;
use std::rc::Rc;

use dypdl::prelude::*;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::search_algorithm::data_structure;

use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;

pub fn pop_from_open<T, N>(
    open: &mut BinaryHeap<Rc<N>>,
    model: &Model,
    primal_bound: Option<T>,
) -> Option<Rc<N>>
where
    T: Numeric + Display,
    N: BfsNodeWithDistributedIdChain<T>,
{
    while let Some(node) = open.pop() {
        if node.is_closed() {
            continue;
        }

        node.close();

        if node.bound(model).map_or(false, |bound| {
            data_structure::exceed_bound(model, bound, primal_bound)
        }) {
            if N::ordered_by_bound() {
                open.clear();

                return None;
            }

            continue;
        }

        return Some(node);
    }

    None
}

pub fn pop_from_open_with_depth<T, N>(
    open: &mut BinaryHeap<(Rc<N>, usize)>,
    model: &Model,
    primal_bound: Option<T>,
) -> Option<(Rc<N>, usize)>
where
    T: Numeric + Display,
    N: BfsNodeWithDistributedIdChain<T>,
{
    while let Some((node, depth)) = open.pop() {
        if node.is_closed() {
            continue;
        }

        node.close();

        if node.bound(model).map_or(false, |bound| {
            data_structure::exceed_bound(model, bound, primal_bound)
        }) {
            if N::ordered_by_bound() {
                open.clear();

                return None;
            }

            continue;
        }

        return Some((node, depth));
    }

    None
}
