"""
Copyright 2025 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# PyTorch translation of the TensorFlow/Keras RouteNet-Gauss model.
#
# Reading guide: every original TensorFlow line is kept, commented out with a `#TF:` prefix,
# and its PyTorch translation follows directly below it. Lines that contain no TensorFlow are
# unchanged. The frozen TF original is importable as `tf_reference.models`. See
# PYTORCH_PORT.md for the op-by-op mapping and the list of semantic differences, and
# torch_ragged.py for the tf.RaggedTensor stand-ins used here (Ragged, ragged_gather, ...).

#TF: import tensorflow as tf
import torch
from torch import nn

from torch_ragged import (
    Ragged,
    ragged_gather,
    ragged_gather_nd,
    ragged_prepend,
    ragged_reduce_sum,
    run_gru_over_ragged,
)


def init_keras_style_(module: nn.Module) -> None:
    """Re-initialise `module` in place with the Keras defaults RouteNet-Gauss was trained
    with: glorot-uniform kernels (Dense and GRU input kernels), orthogonal GRU recurrent
    kernels, zero biases. PyTorch's own defaults differ (kaiming-uniform Linear weights with
    uniform biases; uniform(-1/sqrt(H), 1/sqrt(H)) for every GRU tensor), so this is what
    `RouteNetGauss(init="keras")` uses for the TF-comparison runs. Note that the *numbers*
    still come from PyTorch's RNG — identical initial weights to a TF run are obtained by
    loading them (see tf_reference/replay_tf_run.py and convert_tf_checkpoint.py)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # glorot_uniform: fan_in=in, fan_out=out, same as Keras
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRUCell):
            nn.init.xavier_uniform_(m.weight_ih)  # Keras GRUCell kernel: glorot_uniform over [in, 3H]
            nn.init.orthogonal_(m.weight_hh)  # Keras GRUCell recurrent_kernel: Orthogonal
            nn.init.zeros_(m.bias_ih)  # Keras GRUCell bias [2, 3H]: zeros (reset_after=True)
            nn.init.zeros_(m.bias_hh)
        elif isinstance(m, nn.GRU):  # same cell, sequence form (flow_update)
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            nn.init.zeros_(m.bias_ih_l0)
            nn.init.zeros_(m.bias_hh_l0)


#TF: class RouteNetGauss(tf.keras.Model):
class RouteNetGauss(nn.Module):
    z_scores_fields = {
        "flow_traffic",
        "flow_packets",
    }

    def __init__(
        self,
        z_scores: dict,
        mask_field: str,
        iterations: int = 8,
        flow_state_dim: int = 32,
        link_state_dim: int = 32,
        queue_state_dim: int = 32,
        node_state_dim: int = 32,
        output_dim: int = 1,
        inference_mode: bool = False,
        use_trans_delay: bool = False,
        init: str = "torch",
    ):
        """RouteNet-Gauss model

        Parameters
        ----------
        z_scores : dict
            Z-scores of normalized features. Use self.z_scores_fields to check which
            features are needed.
        mask_field : str,
            Field from the input data that is used to mask out windows without packets
            in a given window. Usually has the form of "flow_has_X", where X is the
            perfomance metric to predict (delay requires at least one packet, jitter
            two).
        iterations : int, optional
            Number of iterations in the Message Passing, by default 8.
        flow_state_dim : int, optional
            Dimension of flow embeddings, by default 32.
        link_state_dim : int, optional
            Dimension of link embeddings, by default 32.
        queue_state_dim : int, optional
            Dimension of queue embeddings, by default 32.
        node_state_dim : int, optional
            Dimension of node embeddings, by default 32.
        output_dim : int, optional
            Number of outputs, by default 1.
        inference_mode : bool, optional
            If true, predictions by the model will be forced to be positive. Doing so
            during training results in poorer learning. By default False.
        use_trans_delay : bool, optional
            If true, transmission delay will be obtained from the measured inputs.
            Useful when aiming to predict the delay, as it allows the model to focus
            only on the queueing delay. By default False
        init : str, optional
            PyTorch-only. "torch" (default) keeps PyTorch's default parameter
            initialisation; "keras" re-initialises with the Keras defaults the TF model
            used (glorot-uniform kernels, orthogonal recurrent kernels, zero biases), which
            is what the TF-comparison experiments pass explicitly.

        Inputs (PyTorch): `forward` takes the same dict of per-scenario tensors as the TF
        `call`, with tf.RaggedTensor fields (`path_to_link`, `link_to_path`,
        `node_groupings`) given as `torch_ragged.Ragged`; `utils.load_dataset` produces
        exactly this from data_torch/.
        """
        super().__init__()

        self.max_buffer_types = 3

        self.iterations = iterations
        self.flow_state_dim = flow_state_dim
        self.link_state_dim = link_state_dim
        self.queue_state_dim = queue_state_dim
        self.node_state_dim = node_state_dim
        self.output_dim = output_dim

        assert mask_field is not None, "mask_field must be specified"
        self.mask_field = mask_field

        self.z_scores = z_scores
        assert (
            type(z_scores) == dict
            and all(kk in self.z_scores for kk in self.z_scores_fields)
            and all(len(val) == 2 for val in self.z_scores.values())
        ), "overriden z_score dict is not valid!"
        # PyTorch: keep the z-scores as buffers so they follow the model's device and the
        # normalisation is computed in the model dtype — float32 by default, exactly like TF
        # did with numpy float32 scalars (`self.z_scores` is kept unchanged for introspection).
        # torch.get_default_dtype() (== tf.float32 in the TF code) is used instead of a
        # hard-coded float32 so the model can be run in float64 for numerical diagnostics.
        for kk in sorted(self.z_scores_fields):
            self.register_buffer(f"z_{kk}_mean", torch.tensor(float(self.z_scores[kk][0]), dtype=torch.get_default_dtype()))
            self.register_buffer(f"z_{kk}_std", torch.tensor(float(self.z_scores[kk][1]), dtype=torch.get_default_dtype()))
        # Calculate the tranmission delay separately (should be used only when
        # predicting the delay)
        self.use_trans_delay = use_trans_delay
        # Force the queuing predictions to be positive
        self.inference_mode = inference_mode

        # GRU Cells used in the Message Passing step
        # PyTorch: nn.GRUCell needs the input size up front (Keras infers it on first call);
        # the gate equations are those of Keras' GRUCell(reset_after=True) — see
        # PYTORCH_PORT.md for the gate-order mapping used when loading TF weights.
        # flow_update is only ever used through tf.keras.layers.RNN(self.flow_update, ...) in
        # call(), i.e. run along each flow's hop sequence; torch's sequence form of the same
        # cell is nn.GRU (one layer, batch_first). Its parameters (weight_ih_l0, ...) have the
        # GRUCell layout, so the TF->torch weight mapping is the same as for the other cells.
        #TF: self.flow_update = tf.keras.layers.GRUCell(
        #TF:     self.flow_state_dim, name="PathUpdate"
        #TF: )
        self.flow_update = nn.GRU(
            self.queue_state_dim + self.link_state_dim, self.flow_state_dim, batch_first=True
        )
        #TF: self.link_update = tf.keras.layers.GRUCell(
        #TF:     self.link_state_dim, name="LinkUpdate"
        #TF: )
        self.link_update = nn.GRUCell(self.queue_state_dim, self.link_state_dim)
        #TF: self.queue_update = tf.keras.layers.GRUCell(
        #TF:     self.queue_state_dim, name="QueueUpdate"
        #TF: )
        self.queue_update = nn.GRUCell(
            self.flow_state_dim + self.node_state_dim, self.queue_state_dim
        )
        #TF: self.node_update = tf.keras.layers.GRUCell(
        #TF:     self.node_state_dim, name="NodeUpdate"
        #TF: )
        self.node_update = nn.GRUCell(self.queue_state_dim, self.node_state_dim)

        # Embedding functions
        # PyTorch: nn.Sequential has no Input layer; Keras Dense(units, activation=relu) is
        # nn.Linear followed by nn.ReLU, so TF layer_with_weights-k is index 2k here.
        #TF: self.flow_embedding = tf.keras.Sequential(
        #TF:     [
        #TF:         tf.keras.layers.Input(shape=(None, 3)),
        #TF:         tf.keras.layers.Dense(
        #TF:             self.flow_state_dim, activation=tf.keras.activations.relu
        #TF:         ),
        #TF:         tf.keras.layers.Dense(
        #TF:             self.flow_state_dim, activation=tf.keras.activations.relu
        #TF:         ),
        #TF:     ],
        #TF:     name="PathEmbedding",
        #TF: )
        self.flow_embedding = nn.Sequential(
            nn.Linear(3, self.flow_state_dim),
            nn.ReLU(),
            nn.Linear(self.flow_state_dim, self.flow_state_dim),
            nn.ReLU(),
        )
        #TF: self.queue_embedding = tf.keras.Sequential(
        #TF:     [
        #TF:         tf.keras.layers.Input(shape=self.max_buffer_types),
        #TF:         tf.keras.layers.Dense(
        #TF:             self.queue_state_dim, activation=tf.keras.activations.relu
        #TF:         ),
        #TF:         tf.keras.layers.Dense(
        #TF:             self.queue_state_dim, activation=tf.keras.activations.relu
        #TF:         ),
        #TF:     ],
        #TF:     name="QueueEmbedding",
        #TF: )
        self.queue_embedding = nn.Sequential(
            nn.Linear(self.max_buffer_types, self.queue_state_dim),
            nn.ReLU(),
            nn.Linear(self.queue_state_dim, self.queue_state_dim),
            nn.ReLU(),
        )
        #TF: self.link_embedding = tf.keras.Sequential(
        #TF:     [
        #TF:         tf.keras.layers.Input(shape=(None, 1)),
        #TF:         tf.keras.layers.Dense(
        #TF:             self.link_state_dim, activation=tf.keras.activations.relu
        #TF:         ),
        #TF:         tf.keras.layers.Dense(
        #TF:             self.link_state_dim, activation=tf.keras.activations.relu
        #TF:         ),
        #TF:     ],
        #TF:     name="LinkEmbedding",
        #TF: )
        self.link_embedding = nn.Sequential(
            nn.Linear(1, self.link_state_dim),
            nn.ReLU(),
            nn.Linear(self.link_state_dim, self.link_state_dim),
            nn.ReLU(),
        )
        #TF: self.node_embedding = tf.keras.Sequential(
        #TF:     [
        #TF:         tf.keras.layers.Input(shape=self.queue_state_dim),
        #TF:         tf.keras.layers.Dense(
        #TF:             (self.queue_state_dim + self.node_state_dim) // 2,
        #TF:             activation=tf.keras.activations.relu,
        #TF:         ),
        #TF:         tf.keras.layers.Dense(
        #TF:             self.node_state_dim, activation=tf.keras.activations.relu
        #TF:         ),
        #TF:     ]
        #TF: )
        self.node_embedding = nn.Sequential(
            nn.Linear(self.queue_state_dim, (self.queue_state_dim + self.node_state_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.queue_state_dim + self.node_state_dim) // 2, self.node_state_dim),
            nn.ReLU(),
        )

        #TF: self.readout_path = tf.keras.Sequential(
        #TF:     [
        #TF:         tf.keras.layers.Input(shape=(None, self.flow_state_dim)),
        #TF:         tf.keras.layers.Dense(
        #TF:             int(self.link_state_dim / 2), activation=tf.keras.activations.relu
        #TF:         ),
        #TF:         tf.keras.layers.Dense(
        #TF:             int(self.flow_state_dim / 2), activation=tf.keras.activations.relu
        #TF:         ),
        #TF:         tf.keras.layers.Dense(self.output_dim),
        #TF:     ],
        #TF:     name="PathReadout",
        #TF: )
        self.readout_path = nn.Sequential(
            nn.Linear(self.flow_state_dim, int(self.link_state_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(self.link_state_dim / 2), int(self.flow_state_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(self.flow_state_dim / 2), self.output_dim),
        )

        # PyTorch-only: parameter initialisation scheme (see docstring).
        if init == "keras":
            init_keras_style_(self)
        elif init != "torch":
            raise ValueError(f"init must be 'torch' or 'keras', got {init!r}")
        self.init = init

    #TF: @tf.function
    #TF: def call(self, inputs):
    # PyTorch runs eagerly; there is no graph tracing (and hence no per-topology retracing).
    def forward(self, inputs):
        device = inputs["flow_traffic"].device
        # Initialize result matrix
        #TF: total_delay = tf.zeros((0, self.output_dim))
        total_delay = torch.zeros((0, self.output_dim), device=device)

        #TF: seg_num = inputs["seg_num"]
        seg_num = int(inputs["seg_num"])
        # PyTorch: index tensors are int32 on disk (as in TF); indexing needs int64.
        #TF: flow_to_link = flow_to_queue = inputs["path_to_link"]
        flow_to_link = flow_to_queue = inputs["path_to_link"].long()
        #TF: node_groupings = inputs["node_groupings"]
        node_groupings = inputs["node_groupings"].long()
        #TF: inverse_node_groupings = inputs["node_groupings_inversed"]
        inverse_node_groupings = inputs["node_groupings_inversed"].long()
        #TF: queue_to_link = inputs["queue_to_link"]
        queue_to_link = inputs["queue_to_link"].long()
        #TF: link_to_path = queue_to_path = inputs["link_to_path"]
        link_to_path = queue_to_path = inputs["link_to_path"].long()

        # Initial embeddings
        traffic = inputs["flow_traffic"]
        pkt_rate = inputs["flow_packets"]
        pkt_size = inputs["flow_packet_size"]
        #TF: length = tf.squeeze(inputs["flow_length"], 1)
        length = inputs["flow_length"].squeeze(1)
        flow_has_traffic = inputs["flow_has_traffic"]
        # We apply the transpose so the first dimension are the segments, the second the
        # flows
        #TF: initial_flow_state = tf.transpose(
        #TF:     self.flow_embedding(
        #TF:         tf.concat(
        #TF:             [
        #TF:                 (traffic - self.z_scores["flow_traffic"][0])
        #TF:                 / self.z_scores["flow_traffic"][1],
        #TF:                 (pkt_rate - self.z_scores["flow_packets"][0])
        #TF:                 / self.z_scores["flow_packets"][1],
        #TF:                 tf.expand_dims(tf.cast(flow_has_traffic, tf.float32), 2),
        #TF:             ],
        #TF:             axis=2,
        #TF:         ),
        #TF:     ),
        #TF:     perm=[1, 0, 2],
        #TF: )
        initial_flow_state = self.flow_embedding(
            torch.cat(
                [
                    (traffic - self.z_flow_traffic_mean) / self.z_flow_traffic_std,
                    (pkt_rate - self.z_flow_packets_mean) / self.z_flow_packets_std,
                    flow_has_traffic.to(torch.get_default_dtype()).unsqueeze(2),
                ],
                dim=2,
            ),
        ).permute(1, 0, 2)

        # Calculate load per link per window, including packet size correction due to
        # l1 and l2 headers size
        if "link_capacity" not in inputs:
            #TF: capacity = (
            #TF:     tf.concat(
            #TF:         [inputs["link_r_capacity"], inputs["link_s_capacity"]], axis=0
            #TF:     )
            #TF:     * 1e9
            #TF: )
            capacity = (
                torch.cat(
                    [inputs["link_r_capacity"], inputs["link_s_capacity"]], dim=0
                )
                * 1e9
            )
        else:
            capacity = inputs["link_capacity"] * 1e9
        #TF: expanded_capacity = tf.tile(tf.expand_dims(capacity, 1), [1, seg_num, 1])
        expanded_capacity = capacity.unsqueeze(1).repeat(1, seg_num, 1)
        if "link_pkt_header_size" not in inputs:
            #TF: pkt_size_correction = tf.concat(
            #TF:     [inputs["link_r_pkt_header_size"], inputs["link_s_pkt_header_size"]],
            #TF:     axis=0,
            #TF: )
            pkt_size_correction = torch.cat(
                [inputs["link_r_pkt_header_size"], inputs["link_s_pkt_header_size"]],
                dim=0,
            )
        else:
            pkt_size_correction = inputs["link_pkt_header_size"]
        #TF: pkt_size_correction = tf.tile(
        #TF:     tf.expand_dims(pkt_size_correction, 1), [1, seg_num, 1]
        #TF: )
        pkt_size_correction = pkt_size_correction.unsqueeze(1).repeat(1, seg_num, 1)
        #TF: flow_gather_traffic = tf.gather(traffic, flow_to_link[:, :, 0])
        flow_gather_traffic = ragged_gather(traffic, flow_to_link.with_values(flow_to_link.values[:, 0]))
        #TF: flow_traffic = tf.math.reduce_sum(flow_gather_traffic, axis=1)
        flow_traffic = ragged_reduce_sum(flow_gather_traffic)
        #TF: flow_gather_pkt_rate = tf.gather(pkt_rate, flow_to_link[:, :, 0])
        flow_gather_pkt_rate = ragged_gather(pkt_rate, flow_to_link.with_values(flow_to_link.values[:, 0]))
        #TF: flow_pkt_rate = tf.math.reduce_sum(flow_gather_pkt_rate, axis=1)
        flow_pkt_rate = ragged_reduce_sum(flow_gather_pkt_rate)
        load = (flow_traffic + flow_pkt_rate * pkt_size_correction) / expanded_capacity
        # We apply the transpose so the first dimension are the segments, the second the
        # links
        #TF: initial_link_state = tf.transpose(self.link_embedding(load), [1, 0, 2])
        initial_link_state = self.link_embedding(load).permute(1, 0, 2)

        # Queue_state and node states are related to memory buffers, these are the
        # states that are kept between windows
        buffer_type = inputs["buffer_type"]
        #TF: queue_state = self.queue_embedding(
        #TF:     tf.squeeze(tf.one_hot(buffer_type, self.max_buffer_types), 1)
        #TF: )
        queue_state = self.queue_embedding(
            torch.nn.functional.one_hot(buffer_type.long(), self.max_buffer_types).squeeze(1).to(torch.get_default_dtype())
        )
        #TF: node_state = self.node_embedding(
        #TF:     tf.math.reduce_sum(
        #TF:         tf.gather(queue_state, node_groupings),
        #TF:         axis=1,
        #TF:         name="RQueueGrouping-Embedding",
        #TF:     ),
        #TF: )
        node_state = self.node_embedding(
            ragged_reduce_sum(
                ragged_gather(queue_state, node_groupings),
            ),
        )

        # Variables for tf.autograd
        #TF: flow_state_sequence = tf.RaggedTensor.from_row_lengths(
        #TF:     tf.zeros((tf.reduce_sum(length), self.flow_state_dim)), length
        #TF: ).with_row_splits_dtype(tf.int64)
        flow_state_sequence = Ragged.from_row_lengths(
            torch.zeros((int(length.sum()), self.flow_state_dim), device=device), length
        )

        #TF: for curr_seg in range(inputs["seg_num"]):
        for curr_seg in range(seg_num):
            # PyTorch: autograph loop hints have no equivalent (plain Python loop).
            #TF: tf.autograph.experimental.set_loop_options(
            #TF:     shape_invariants=[
            #TF:         (total_delay, tf.TensorShape([None, self.output_dim])),
            #TF:         (
            #TF:             flow_state_sequence,
            #TF:             tf.TensorShape([None, None, self.flow_state_dim]),
            #TF:         ),
            #TF:     ],
            #TF: )

            # Initialize segment states for flows and links
            flow_state = initial_flow_state[curr_seg]
            link_state = initial_link_state[curr_seg]

            # Iterate t times doing the message passing
            for it in range(self.iterations):
                ###################
                #  LINK AND QUEUE #
                #     TO PATH     #
                ###################
                #TF: queue_gather = tf.gather(queue_state, queue_to_path)
                queue_gather = ragged_gather(queue_state, queue_to_path)
                #TF: link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
                link_gather = ragged_gather(link_state, link_to_path)
                # PyTorch: the Keras RNN wrapper (sequence run of the cell with masking for
                # the ragged rows) is run_gru_over_ragged in torch_ragged.py.
                #TF: flow_update_rnn = tf.keras.layers.RNN(
                #TF:     self.flow_update, return_sequences=True, return_state=True
                #TF: )
                previous_flow_state = flow_state

                # flow_state -> state of path after processing sequence
                # flow_state_sequence -> sequence of intermediate states of the path
                # when elements within the sequence are processed
                #TF: flow_state_sequence, flow_state = flow_update_rnn(
                #TF:     tf.concat([queue_gather, link_gather], axis=2),
                #TF:     initial_state=flow_state,
                #TF: )
                flow_state_sequence, flow_state = run_gru_over_ragged(
                    self.flow_update,
                    link_gather.with_values(torch.cat([queue_gather.values, link_gather.values], dim=1)),
                    flow_state,
                )
                # We select the element in flow_state_sequence so that it corresponds to
                # the state before the link was considered
                #TF: flow_state_sequence = tf.concat(
                #TF:     [tf.expand_dims(previous_flow_state, 1), flow_state_sequence],
                #TF:     axis=1,
                #TF: )
                flow_state_sequence = ragged_prepend(previous_flow_state, flow_state_sequence)

                ###################
                #  PATH AND NODE  #
                #    TO QUEUE     #
                ###################
                #TF: flow_gather = tf.gather_nd(flow_state_sequence, flow_to_queue)
                flow_gather = ragged_gather_nd(flow_state_sequence.to_padded(), flow_to_queue)
                #TF: flow_sum = tf.math.reduce_sum(flow_gather, axis=1)
                flow_sum = ragged_reduce_sum(flow_gather)
                #TF: node_gather = tf.gather(
                #TF:     node_state, inverse_node_groupings, name="NodeStateUnfolded"
                #TF: )
                node_gather = node_state[inverse_node_groupings]

                #TF: queue_state, _ = self.queue_update(
                #TF:     tf.concat([flow_sum, node_gather], axis=1), [queue_state]
                #TF: )
                queue_state = self.queue_update(
                    torch.cat([flow_sum, node_gather], dim=1), queue_state
                )

                ###################
                #  QUEUE TO LINK  #
                ###################
                #TF: queue_gather = tf.gather(queue_state, queue_to_link)
                queue_gather = queue_state[queue_to_link]

                # PyTorch: queue_to_link is [n_links, 1], so the RNN runs over a length-1
                # sequence, i.e. exactly one GRU cell step.
                #TF: link_gru_rnn = tf.keras.layers.RNN(
                #TF:     self.link_update, return_sequences=False
                #TF: )
                #TF: link_state = link_gru_rnn(queue_gather, initial_state=link_state)
                link_state = self.link_update(queue_gather[:, 0], link_state)

                ###################
                #  QUEUE TO NODE  #
                ###################
                #TF: node_state, _ = self.node_update(
                #TF:     tf.math.reduce_sum(
                #TF:         tf.gather(queue_state, node_groupings),
                #TF:         axis=1,
                #TF:         name="NodeQueueGrouping",
                #TF:     ),
                #TF:     states=node_state,
                #TF: )
                node_state = self.node_update(
                    ragged_reduce_sum(
                        ragged_gather(queue_state, node_groupings),
                    ),
                    node_state,
                )

            ###################
            # MESSAGE PASSING #
            #       END       #
            ###################
            # Readout and delay prediction
            #TF: capacity_gather = tf.gather(capacity, link_to_path)
            capacity_gather = ragged_gather(capacity, link_to_path)
            #TF: input_tensor = flow_state_sequence[:, 1:].to_tensor()
            input_tensor = flow_state_sequence.inner_slice_from(1).to_padded()

            occupancy_gather = self.readout_path(input_tensor)
            #TF: length = tf.ensure_shape(length, [None])
            #TF: occupancy_gather = tf.RaggedTensor.from_tensor(
            #TF:     occupancy_gather, lengths=length
            #TF: )
            occupancy_gather = Ragged.from_padded(occupancy_gather, length)

            #TF: queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather, axis=1)
            queue_delay = ragged_reduce_sum(occupancy_gather.with_values(occupancy_gather.values / capacity_gather.values))
            if self.use_trans_delay:
                #TF: trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)
                trans_delay = pkt_size * ragged_reduce_sum(capacity_gather.with_values(1 / capacity_gather.values))
                delay = queue_delay + trans_delay
            else:
                delay = queue_delay
            if self.inference_mode:
                #TF: delay = tf.keras.activations.relu(delay)
                delay = torch.relu(delay)

            #TF: delay = tf.boolean_mask(delay, inputs[self.mask_field][:, curr_seg])
            delay = delay[inputs[self.mask_field][:, curr_seg]]

            #TF: total_delay = tf.concat([total_delay, delay], axis=0)
            total_delay = torch.cat([total_delay, delay], dim=0)

        ##################
        #     WINDOW     #
        # PROCESSING END #
        ##################

        return total_delay
