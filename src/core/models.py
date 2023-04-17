from matplotlib.pyplot import new_figure_manager
from multiprocessing import connection
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Sequential, global_add_pool, JumpingKnowledge
from torchvision.models._utils import IntermediateLayerGetter
from math import floor
import torch
from torchsummary import summary
import numpy as np


class MLP(nn.Module):
    """
    Two-layer MLP network
    # TODO: Add link to MICCAI paper

    Attributes
    ----------
    fc_1: torch.nn.Module, first FC linear layer
    fc_2: torch.nn.Module, second FC linear layer
    bn: torch.nn.Module, batch normalization layer
    dropout_p: float, dropout ratio

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 80,
                 output_dim: int = 128,
                 dropout_p: int = 0.0):
        """
        :param input_dim: int, dimension of input embeddings
        :param hidden_dim: int, dimension of hidden embeddings
        :param output_dim: int, dimension of output embeddings
        :param dropout_p: float, dropout used in between layers
        """

        super().__init__()

        # Linear layers
        self.fc_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc_2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        # Initialize batch norm
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.bn_out = nn.BatchNorm1d(output_dim)

        # Dropout params
        self.dropout_p = dropout_p

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation

        :param x: torch.tensor, input tensor
        :return: transformed embeddings
        """

        # Two FC layers
        x = F.relu(self.bn_out(self.fc_2(F.dropout(F.relu(self.bn(self.fc_1(x))),
                                                   p=self.dropout_p,
                                                   training=self.training))))

        return x


class CNNResBlock(nn.Module):
    """
    3D convolution block with residual connections
    #TODO: Add link to code for MICCAI paper

    Attributes
    ----------
    conv: torch.nn.Conv3d, PyTorch Conv3D model
    bn: torch.nn.BatchNorm3d, PyTorch 3D batch normalization layer
    pool: torch.nn.AvgPool3d, PyTorch average 3D pooling layer
    dropout: torch.nn.Dropout3D, PyTorch 3D dropout layer
    one_by_one_cnn: torch.nn.Conv3d, pyTorch 1*1 conv model to equalize the number of channels for residual addition

    Methods
    -------
    forward(x): model's forward propagation
    """
    def __init__(self,
                 in_channels: int,
                 padding: int,
                 out_channels: int = 128,
                 kernel_size: int = 3,
                 pool_size: int = 2,
                 out_size: int = None,
                 cnn_dropout_p: float = 0.0):
        """
        :param in_channels: int, number of input channels
        :param padding: int, 0 padding dims
        :param out_channels: int, number of filters to use
        :param kernel_size: int, filter size
        :param pool_size: int, pooling kernel size for the spatial dims (if out_size is kept as None)
        :param out_size: int, output frame dimension for adaptive pooling
        :param cnn_dropout_p: float, cnn dropout rate
        """

        super().__init__()

        # Check if a Conv would be needed to make the channel dim the same
        # for the residual
        self.one_by_one_cnn = None
        if in_channels != out_channels:
            # noinspection PyTypeChecker
            self.one_by_one_cnn = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1)

        # 2D conv layer
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=(padding, padding))

        # Other operations
        self.bn = nn.BatchNorm2d(out_channels)
        if out_size is None:
            self.pool = nn.MaxPool2d(kernel_size=(pool_size, pool_size))
        else:
            self.pool = nn.AdaptiveMaxPool2d(output_size=(out_size, out_size))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(p=cnn_dropout_p)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation

        :param x: torch.tensor, input tensor of shape N*1*64*T*H*W
        :return: Tensor of shape N*out_channels*T*H'*W'
        """

        # Make the number of channels equal for input and output if needed
        if self.one_by_one_cnn is not None:
            residual = self.one_by_one_cnn(x)
        else:
            residual = x

        x = self.conv(x)
        x = self.bn(x)
        x = x + residual
        x = self.pool(x)
        x = self.activation(x)

        return self.dropout(x)


class CNN(nn.Module):
    """
    3D convolution network
    # TODO: Add link to MICCAI paper

    Attributes
    ----------
    conv: torch.nn.Sequential, the convolutional network containing residual blocks
    output_fc: torch.nn.Sequential, the FC layer applied to the output of convolutional network

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 out_channels: list,
                 kernel_sizes: list = None,
                 pool_sizes: list = None,
                 fc_output_dim: list = None,
                 cnn_dropout_p: float = 0.0):
        """
        :param out_channels: list, output channels for each layer
        :param kernel_sizes: list, kernel sizes for each layer
        :param pool_sizes: list, pooling kernel sizes for each layer
        :param fc_output_dim: int, the output dimension of output FC layer (set to None for no output fc)
        :param cnn_dropout_p: float, dropout ratio of the CNN
        """

        super().__init__()

        n_conv_layers = len(out_channels)

        # Default list arguments
        if kernel_sizes is None:
            kernel_sizes = [3]*n_conv_layers
        if pool_sizes is None:
            pool_sizes = [1]*n_conv_layers

        # Ensure input params are list
        if type(out_channels) is not list:
            out_channels = [out_channels]*n_conv_layers
        else:
            assert len(out_channels) == n_conv_layers, 'Provide channel parameter for all layers.'
        if type(kernel_sizes) is not list:
            kernel_sizes = [kernel_sizes]*n_conv_layers
        else:
            assert len(kernel_sizes) == n_conv_layers, 'Provide kernel size parameter for all layers.'
        if type(pool_sizes) is not list:
            pool_sizes = [pool_sizes]*n_conv_layers
        else:
            assert len(pool_sizes) == n_conv_layers, 'Provide pool size parameter for all layers.'

        # Compute paddings to preserve temporal dim
        paddings = list()
        for kernel_size in kernel_sizes:
            paddings.append(floor((kernel_size - 1) / 2))

        # Conv layers
        convs = list()

        # Add first layer
        convs.append(nn.Sequential(CNNResBlock(in_channels=1,
                                               padding=paddings[0],
                                               out_channels=out_channels[0],
                                               kernel_size=kernel_sizes[0],
                                               pool_size=pool_sizes[0],
                                               cnn_dropout_p=cnn_dropout_p)))

        # Add subsequent layers
        for layer_num in range(1, n_conv_layers):
            convs.append(nn.Sequential(CNNResBlock(in_channels=out_channels[layer_num-1],
                                                   padding=paddings[layer_num],
                                                   out_channels=out_channels[layer_num],
                                                   kernel_size=kernel_sizes[layer_num],
                                                   pool_size=pool_sizes[layer_num],
                                                   cnn_dropout_p=cnn_dropout_p)))
        # Change to sequential
        self.conv = nn.Sequential(*convs)

        # Output linear layer
        self.output_fc = None
        if fc_output_dim is not None:
            self.output_fc = nn.Sequential(nn.AdaptiveAvgPool3d((None, 1, 1)),
                                           nn.Flatten(start_dim=2),
                                           nn.Linear(out_channels[-1], fc_output_dim),
                                           nn.ReLU(inplace=True))

    def forward(self,
                x):
        """
        Forward path of the CNN3D network

        :param x: torch.tensor, input torch.tensor of image frames

        :return: Vector embeddings of input images of shape (num_samples, output_dim)
        """

        # CNN layers
        x = self.conv(x)

        # FC layer
        if self.output_fc is not None:
            x = self.output_fc(x)

        return x


class HierarchicalPatchModel(nn.Module):
    """
    Hierarchical landmark detection model using GNNs. This model create multiple auxiliary graphs
    from a given frame and performs landmark detection for each graph.

    Attributes
    ----------
    vn_embedding: torch.nn.Embedding, virtual node embeddings allowing communication between graphs
    gnn_layers: torch_geometric.nn.Sequential, GNN layers
    vn_mlps: torch.nn.Sequential, MLPs used to transform the VN embeddings
    node_classifier: torch.nn.Sequential, classifier used to predict each node's class (either a landmark or not)
    jk: torch_geometric.nn.JumpingKnowledge, Jumping knowledge module
    frame_size: int, input frame's dimension
    residual: bool, indicates whether a residual connection is used
    num_gnn_layers: int, number of GNN layers
    node_embedding_dim: int, the hidden dimension of node embeddings
    num_aux_graphs: int, number of auxilary graphs

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 frame_size: int= 32,
                 gnn_dropout_p: float = 0.0,
                 classifier_dropout_p: float = 0.0,
                 node_embedding_dim: int = 128,
                 node_hidden_dim: int = 64,
                 num_output_channels: int = 4,
                 num_gnn_layers: int = 3,
                 num_aux_graphs: int = 4,
                 gnn_jk_mode: str = 'last',
                 classifier_hidden_dim: int = 16,
                 residual: bool = True,
                 use_coordinate_graph: bool = False,
                 output_activation: str = 'sigmoid',
                 use_connection_nodes=False,
                 use_main_graph_only=False):
        """
        :param frame_size: int, input frame dimension
        :param gnn_dropout_p: float, the dropout layer for GNN layers
        :param classifier_dropout_p: float, the dropout for the classifier MLP
        :param node_embedding_dim: int, the node input embedding dim
        :param node_hidden_dim: int, the node hidden embedding dim
        :param num_output_channels: int, number of node classifiers
        :param num_gnn_layers: int, number of GNN layers
        :param num_aux_graphs: int, number of auxiliary graphs to create
        :param gnn_jk_mode: str, jumping knowledge mode to use. must be one of 'last', 'max', 'cat'
        :param classifier_hidden_dim: int, the hidden dimension of classifier MLP
        :param use_coordinate_graph: bool, indicates whether a coordinate main graph is used
        :param residual: bool, indicates whether residual connections are added
        :param use_connection_nodes: bool, indicates whether connection nodes are used
        :param use_main_graph_only: bool, Used for ablation study when only the main graph exists
        """

        super().__init__()

        # Check valid JK is requested
        assert gnn_jk_mode in ['last', 'max', 'cat'], "Only last, max or cat jumping knowledge mode is supported."

        # Create the GNN network
        self.gnn_layers = nn.ModuleList()
        self.node_coordinate_mlp = nn.ModuleList()

        for i in range(num_gnn_layers):
            self.gnn_layers.append(Sequential('x, edge_index', [
                (GCNConv(in_channels=node_embedding_dim if i == 0 else node_hidden_dim,
                         out_channels=node_hidden_dim),
                 'x, edge_index -> x'),
                nn.BatchNorm1d(node_hidden_dim),
                nn.Dropout(p=gnn_dropout_p),
                nn.Identity() if i == num_gnn_layers-1 else nn.ReLU(inplace=True)]))

            # Coordinate regressor
            if use_coordinate_graph:
                self.node_coordinate_mlp.append(nn.Sequential(nn.Linear(in_features=node_hidden_dim + 8,
                                                                        out_features=classifier_hidden_dim),
                                                              nn.BatchNorm1d(classifier_hidden_dim),
                                                              nn.ReLU(inplace=True),
                                                              nn.Dropout(p=classifier_dropout_p),
                                                              nn.Linear(in_features=classifier_hidden_dim,
                                                                        out_features=classifier_hidden_dim // 2),
                                                              nn.BatchNorm1d(classifier_hidden_dim // 2),
                                                              nn.ReLU(inplace=True),
                                                              nn.Dropout(p=classifier_dropout_p),
                                                              nn.Linear(in_features=classifier_hidden_dim // 2,
                                                                        out_features=2),
                                                              nn.Identity()))

        # output activation layer
        self.output_activation = output_activation
        if output_activation == 'sigmoid':
            last_activation_layer = nn.Sigmoid()
        elif output_activation == 'logit':
            last_activation_layer = nn.Identity()
        else:
            raise(f"invalid output_activation:{output_activation}")

        # Create node classifier
        self.node_classifiers = nn.ModuleList()
        for i in range(num_output_channels):
            self.node_classifiers.append(nn.Sequential(nn.Linear(in_features=node_hidden_dim,
                                                                 out_features=classifier_hidden_dim),
                                                       nn.BatchNorm1d(classifier_hidden_dim),
                                                       nn.ReLU(inplace=True),
                                                       nn.Dropout(p=classifier_dropout_p),
                                                       nn.Linear(in_features=classifier_hidden_dim,
                                                                 out_features=classifier_hidden_dim // 2),
                                                       nn.BatchNorm1d(classifier_hidden_dim // 2),
                                                       nn.ReLU(inplace=True),
                                                       nn.Dropout(p=classifier_dropout_p),
                                                       nn.Linear(in_features=classifier_hidden_dim // 2,
                                                                 out_features=1),
                                                       last_activation_layer))

        # Jumping knowledge module
        self.jk = None
        if gnn_jk_mode != 'last':
            self.jk = JumpingKnowledge(mode=gnn_jk_mode)

        # Other attributes
        self.frame_size = frame_size
        self.residual = residual
        self.num_gnn_layers = num_gnn_layers
        self.node_embedding_dim = node_embedding_dim
        self.num_aux_graphs = num_aux_graphs
        self.use_coordinate_graph = use_coordinate_graph
        self.use_connection_nodes = use_connection_nodes
        self.use_main_graph_only = use_main_graph_only

    def forward(self,
                data_batch = None,
                x: torch.tensor = None,
                node_coords: torch.tensor = None,
                edge_index: torch.tensor = None,
                node_type: np.ndarray = None,
                batch_idx: torch.tensor = None) -> torch.tensor:
        """
        Model's forward propagation

        :param data_batch: PyG data batch, Pytorch geometric data batch
        :return: classification predictions for each node
        """

        if data_batch is not None:
            x, edge_index, batch_idx, node_type = data_batch.x, data_batch.edge_index, \
                                                  data_batch.batch, data_batch.node_type

            if self.use_coordinate_graph:
                node_coords = data_batch.node_coords

        if self.use_coordinate_graph:
            node_coords = node_coords.view(node_coords.shape[0] // 4, 4, -1)
        else:
            node_coords = None

        num_samples_in_batch = batch_idx[-1] + 1

        # Create node embeddings from frames
        node_feats = self.create_node_pixels(x, num_samples_in_batch, node_coords)

        # Store layer embeddings in a list for JK
        hidden_embeds = [node_feats]

        for i in range(self.num_gnn_layers):

            # GNN layer
            h = self.gnn_layers[i](hidden_embeds[i], edge_index)

            # Residual identity connection
            if self.residual and h.shape[1] == hidden_embeds[i].shape[1]:
                h = h + hidden_embeds[i]

            # Update node coordinates if needed
            if self.use_coordinate_graph:

                # Create relative distance features
                for j, node_coord in enumerate(node_coords):
                    shape_feats = torch.cat((shape_feats, -1 * (node_coord.unsqueeze(1) - node_coord))) if j != 0 else \
                        -1 * (node_coord.unsqueeze(1) - node_coord)
                shape_feats = shape_feats.flatten(start_dim=1)

                # Combine relative distance features with visual features
                landmark_feats = torch.cat((h[np.where(node_type.detach().cpu().numpy() == 1)[0]], shape_feats), dim=1)

                coords_delta = self.node_coordinate_mlp[i](landmark_feats)
                node_coords += coords_delta.view(coords_delta.shape[0] // 4, 4, -1)

                # Clamp new coords
                node_coords = torch.clamp(node_coords, min=0, max=self.frame_size-1)

                # Need the main graph features to perform bilinear interpolation on
                main_graph_feats = h[np.where(node_type.detach().cpu().numpy() == 0)[0]]
                main_graph_feats = main_graph_feats.view(num_samples_in_batch,
                                                         torch.div(main_graph_feats.shape[0],
                                                                   num_samples_in_batch, rounding_mode='trunc'), -1)
                main_graph_feats = main_graph_feats[:, -self.frame_size * self.frame_size:, :]
                main_graph_feats = main_graph_feats.permute(0, 2, 1)
                main_graph_feats = main_graph_feats.view(main_graph_feats.shape[0],
                                                         main_graph_feats.shape[1], self.frame_size, self.frame_size)

                # Interpolate new features based on new coords
                for i in range(num_samples_in_batch):
                    new_feats = torch.cat((new_feats,
                                           self.bilinear_interpolation(node_coords[i],
                                                                       main_graph_feats[i])), dim=0) \
                        if i != 0 else self.bilinear_interpolation(node_coords[i], main_graph_feats[i])

                # update node embeddings
                h[np.where(node_type.detach().cpu().numpy() == 1)[0]] = new_feats

            # Keep all hidden embeddings in a list for JK
            hidden_embeds.append(h)

        # Jumping knowledge
        if self.jk is not None:
            h = self.jk(hidden_embeds)
        else:
            h = hidden_embeds[-1]

        # Remove connection node embeddings
        h = h[np.where(node_type.detach().cpu().numpy() == 0)[0]]

        # Classify nodes for each channel seperately
        h_out = self.node_classifiers[0](h)
        for i in range(1, len(self.node_classifiers)):
            h_out = torch.cat([h_out, self.node_classifiers[i](h)], dim=1)

        # Reshape node coords to match labels shape
        if self.use_coordinate_graph:
            node_coords = node_coords.view(node_coords.shape[0]*node_coords.shape[1], -1)

        return h_out.squeeze(1), node_coords

    def create_node_pixels(self,
                           echo_frames: torch.tensor,
                           num_samples_per_batch: int,
                           node_coords=None) -> torch.tensor:
        """
        Creates the nodes for the overall graph containing both auxiliary and main graphs

        :param echo_frames: torch.tensor, input image frames
        :param num_samples_per_batch: int, number of samples per batch (number of frames)
        :return: Tensor of shape num_nodes*embedding_dim for the overall graph
        """

        # combining in order of [(2,2), (4,4), (8,8), (16,16), ..., (224,224)]
        all_x = []
        for i in range(num_samples_per_batch):
            x = torch.tensor([]).to(echo_frames[i].device)

            if not self.use_main_graph_only:
                for graph_num in range(1, self.num_aux_graphs+1):
                    x = torch.cat([x, torch.reshape(F.adaptive_avg_pool2d(echo_frames[i],
                                                                          output_size=(2 ** graph_num,
                                                                                       2 ** graph_num)).permute(1,
                                                                                                                2,
                                                                                                                0),
                                                    (-1, self.node_embedding_dim))], dim=0)

            x = torch.cat([x, torch.reshape(echo_frames[i].permute(1, 2, 0), (-1, self.node_embedding_dim))], dim=0)

            if self.use_coordinate_graph:
                x = torch.cat([x, self.bilinear_interpolation(node_coords[i], echo_frames[i])], dim=0)

            # Connection node initial embedding is average of all pixel values
            if self.use_connection_nodes:
                connection_node_embed = torch.reshape(echo_frames[i].mean(dim=(1, 2)), (-1, self.node_embedding_dim))
                connection_node_embed = connection_node_embed.repeat(self.num_aux_graphs + 1, 1)
                x = torch.cat([connection_node_embed, x], dim=0)

            all_x.append(x)

        return torch.cat([x for x in all_x], dim=0)

    def bilinear_interpolation(self, coords, frame):
        coords = coords.T

        w_dist = 1 - torch.abs(coords[1].unsqueeze(1) - torch.arange(start=0, end=frame.shape[-1],
                                                                     device=coords.device))
        w_dist = F.relu(w_dist, inplace=True).unsqueeze(1)

        h_dist = 1 - torch.abs(coords[0].unsqueeze(1) - torch.arange(start=0, end=frame.shape[-1],
                                                                     device=coords.device))
        h_dist = F.relu(h_dist, inplace=True).unsqueeze(2)

        x = torch.bmm(h_dist, w_dist).unsqueeze(1) * frame.unsqueeze(0)
        x = x.sum(-1).sum(-1)

        return x


class CNNHierarchicalPatchModel(HierarchicalPatchModel):
    """
    A child module of HierarchicalPatchModel that creates the grids using a down-sampling CNN encoder
    """
    def __init__(self,
                 cnn_layers_out_width: list = None,
                 cnn_dropout_p: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)

        if cnn_layers_out_width is None:
            cnn_layers_out_width = [128, 64, 32, 16, 8, 4, 2]

        # Conv layers
        convs = []
        for out_size in cnn_layers_out_width:
            convs.append(CNNResBlock(in_channels=self.node_embedding_dim,  # 128
                                     padding=1,
                                     out_channels=self.node_embedding_dim,  # 128
                                     kernel_size=3,
                                     out_size=out_size,  # 128, 64, 32, 16, 8, 4, 2
                                     cnn_dropout_p=cnn_dropout_p))
        # Change to sequential
        self.conv = nn.Sequential(*convs)

        num_layers = len(cnn_layers_out_width)
        return_layers = {f"{num_layers-1-aux_graph_idx}": f"{aux_graph_idx}"
                         for aux_graph_idx in range(self.num_aux_graphs)}
        # {'6': "0", #(128,2,2)
        #  '5': "1", #(128,4,4)
        #  '4': "2", #(128,8,8)
        #  '3': "3"} #(128,16,16)
        self.conv_hierarchical_grids = IntermediateLayerGetter(self.conv, return_layers=return_layers)

    def create_node_pixels(self, echo_frames: torch.tensor, num_samples_per_batch: int, node_coords) -> torch.tensor:
        """
        Creates the nodes for the overall graph containing both auxiliary and main graphs

        :param echo_frames: torch.tensor, input image frames (N, node_embedding_dim, frame_size, frame_size)
        :param num_samples_per_batch: int, number of samples per batch (number of frames)
        :return: Tensor of shape num_nodes*embedding_dim for the overall graph
        """

        feature_grids = self.conv_hierarchical_grids(echo_frames)  # a dictionary with keys ("0", "1", "2", "3"),
        # values have shape of (N, 128, W, H)

        # combining in order of [(2,2), (4,4), (8,8), (16,16), ..., (224,224)]
        all_x = []
        for i in range(num_samples_per_batch):
            x = torch.tensor([]).to(echo_frames[i].device)

            if not self.use_main_graph_only:
                connection_node_embed = torch.tensor([]).to(echo_frames[i].device)
                for graph_num in range(self.num_aux_graphs):
                    x = torch.cat([
                        x,
                        torch.reshape(feature_grids[f"{graph_num}"][i].permute(1, 2, 0), (-1, self.node_embedding_dim)),
                    ], dim=0)

                    if self.use_connection_nodes:
                        connection_node_embed = torch.cat([
                            connection_node_embed,
                            torch.reshape(feature_grids[f"{graph_num}"][i].mean(dim=(1, 2)), (-1, self.node_embedding_dim)),
                        ], dim=0)

            x = torch.cat([x, torch.reshape(echo_frames[i].permute(1, 2, 0), (-1, self.node_embedding_dim))], dim=0)

            if self.use_coordinate_graph:
                x = torch.cat([x, self.bilinear_interpolation(node_coords[i], echo_frames[i])], dim=0)

            # Connection node initial embedding is average of all pixel values
            if self.use_connection_nodes:
                connection_node_embed = torch.cat([
                    connection_node_embed,
                    torch.reshape(echo_frames[i].mean(dim=(1, 2)), (-1, self.node_embedding_dim)),
                ], dim=0)
                x = torch.cat([connection_node_embed, x], dim=0)

            all_x.append(x)

        return torch.cat([x for x in all_x], dim=0)


class UNETHierarchicalPatchModel(HierarchicalPatchModel):
    """
    A child module of HierarchicalPatchModel that creates the grids using a UNET
    where node fetures would be taken from the decoder part of the UNET
    """
    def __init__(self,
                 encoder_embedding_widths: list = None,  # default is for 224x224 image
                 encoder_embedding_dims = None,  # default is for 224x224 image
                 **kwargs):
        super().__init__(**kwargs)

        if encoder_embedding_widths is None:
            encoder_embedding_widths = [128, 64, 32, 16, 8, 4, 2]
        if encoder_embedding_dims:
            encoder_embedding_dims = [8, 16, 32, 64, 128, 256, 512]

        if self.num_aux_graphs > len(encoder_embedding_widths):
            raise(f"num_aux_graphs:{self.num_aux_graphs} is larger "
                  f"than total number of usable intermediate layers:{len(encoder_embedding_widths)}")

        # encoder part, going from (N, 4, 224, 224) to (N, 8, 128, 128) to (N, 512, 2, 2)
        self.down_convs = nn.ModuleList()
        for i, f in enumerate(encoder_embedding_dims):  # [8, 16, 32, 64, 128, 256, 512]
            self.down_convs.append(DownConv(f//2, f, encoder_embedding_widths[i]))

        # decoder part, going from  (N, 512, 2, 2) to (N, 8, 128, 128) to (N, 4, 224, 224)
        self.up_convs = nn.ModuleList()
        decoder_embedding_widths = list(reversed(encoder_embedding_widths))[1:] + [self.frame_size]
        # decoder_embedding_widths = [4, 8, 16, 32, 64, 128, frame_size]
        for i, f in enumerate(reversed(encoder_embedding_dims)):  # [512, 256, 128, 64, 32, 16, 8]
            self.up_convs.append(UpConv(f, f//2, decoder_embedding_widths[i]))
        
        # for chnaging all the channel sized to node_embedding_dim
        in_features = list(reversed(encoder_embedding_dims))  # [512, 256, 128, 64, 32, 16, 8]
        in_features = in_features + [in_features[-1]//2]  # [512, 256, 128, 64, 32, 16, 8, 4]
        out_feature = self.node_embedding_dim
        self.linears = nn.ModuleList()
        for in_f in in_features:
            self.linears.append(nn.Conv2d(in_f, out_feature, kernel_size=1))

    def create_node_pixels(self,
                           echo_frames: torch.tensor,
                           num_samples_per_batch: int,
                           node_coords: torch.tensor = None) -> torch.tensor:
        """
        Creates the nodes for the overall graph containing both auxiliary and main graphs

        :param echo_frames: torch.tensor, input image frames (N, node_embedding_dim, frame_size, frame_size)
        :param num_samples_per_batch: int, number of samples per batch (number of frames)
        :param node_coords: torch.tensor, tensor including landmark coords (N, 4, 2)
        :return: Tensor of shape num_nodes*embedding_dim for the overall graph
        """
        x = echo_frames    # (N, d, frame_size, frame_size)

        skip_connections = []
        for i in range(len(self.down_convs)):
            skip_connections.append(x)
            x = self.down_convs[i](x)
        # last x has shape of (N, 512, 2, 2)

        features = []
        features.append(x)

        for i in range(len(self.up_convs)):  # goes up to (N, d, frame_size, frame_size)
            x = self.up_convs[i](x, skip_connections[-1])
            skip_connections.pop()
            features.append(x)

        # changing all chennel sizes to the same thing
        new_features = []
        for i in range(len(self.linears)):
            new_features.append(F.relu(self.linears[i](features[i])))

        # [torch.Size([10, 128, 2, 2]), 
        # torch.Size([10, 128, 4, 4]), 
        # torch.Size([10, 128, 8, 8]), 
        # torch.Size([10, 128, 16, 16]), 
        # torch.Size([10, 128, 32, 32]), 
        # torch.Size([10, 128, 64, 64]), 
        # torch.Size([10, 128, 128, 128]),
        # torch.Size([10, 128, 224, 224])]
        
        # combining in order of [(2,2), (4,4), (8,8), (16,16), ..., (224,224)]
        all_x = []
        for i in range(num_samples_per_batch):
            x = torch.tensor([]).to(echo_frames[i].device)

            if not self.use_main_graph_only:
                connection_node_embed = torch.tensor([]).to(echo_frames[i].device)
                for graph_num in range(self.num_aux_graphs):
                    x = torch.cat([
                        x,
                        torch.reshape(new_features[graph_num][i].permute(1, 2, 0), (-1, self.node_embedding_dim)),
                    ], dim=0)

                    if self.use_connection_nodes:
                        connection_node_embed = torch.cat([
                            connection_node_embed,
                            torch.reshape(new_features[graph_num][i].mean(dim=(1, 2)), (-1, self.node_embedding_dim)),
                        ], dim=0)

            x = torch.cat([x, torch.reshape(new_features[-1][i].permute(1, 2, 0), (-1, self.node_embedding_dim))]
                             , dim=0)

            if self.use_coordinate_graph:
                x = torch.cat([x, self.bilinear_interpolation(node_coords[i], new_features[-1][i])], dim=0)

            # Connection node initial embedding is average of all pixel values
            if self.use_connection_nodes:
                connection_node_embed = torch.cat([
                    connection_node_embed,
                    torch.reshape(new_features[-1][i].mean(dim=(1, 2)), (-1, self.node_embedding_dim)),
                ], dim=0)
                x = torch.cat([connection_node_embed, x], dim=0)

            all_x.append(x)

        return torch.cat([x for x in all_x], dim=0)


class UNETIntermediateNoGnn(UNETHierarchicalPatchModel):
    """
    A child module of UNETHierarchicalPatchModel that creates the grids using a UNET
    where node features would be taken from the decoder part of the UNET and no GNN is used for node classification
    """

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def forward(self,
                data_batch = None,
                x: torch.tensor = None,
                node_coords: torch.tensor = None,
                edge_index: torch.tensor = None,
                node_type: np.ndarray = None,
                batch_idx: torch.tensor = None) -> torch.tensor:
        """
        Model's forward propagation. Similar to parent's forward function without the GNN portions

        :param data_batch: PyG data batch, Pytorch geometric data batch
        :return: classification predictions for each node
        """

        if data_batch is not None:
            x, edge_index, batch_idx, node_type = data_batch.x, data_batch.edge_index, \
                                                  data_batch.batch, data_batch.node_type

        num_samples_in_batch = batch_idx[-1] + 1

        # Create node embeddings from frames
        h = self.create_node_pixels(x, num_samples_in_batch, node_coords)

        # Remove connection node embeddings
        h = h[np.where(node_type.detach().cpu().numpy() == 0)[0]]

        # Classify nodes for each channel seperately
        h_out = self.node_classifiers[0](h)
        for i in range(1, len(self.node_classifiers)):
            h_out = torch.cat([h_out, self.node_classifiers[i](h)], dim=1)

        return h_out.squeeze(1), None


class UNET(UNETHierarchicalPatchModel):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def forward(self,
                data_batch = None,
                x: torch.tensor = None,
                node_coords: torch.tensor = None,
                edge_index: torch.tensor = None,
                node_type: np.ndarray = None,
                batch_idx: torch.tensor = None) -> torch.tensor:
        """
        Model's forward propagation. Similar to parent's forward function without the GNN portions

        :param data_batch: PyG data batch, Pytorch geometric data batch
        :return: classification predictions for each node
        """

        if data_batch is not None:
            x, edge_index, batch_idx, node_type = data_batch.x, data_batch.edge_index, \
                                                  data_batch.batch, data_batch.node_type

        num_samples_in_batch = batch_idx[-1] + 1

        # Create node embeddings from frames
        h = self.create_node_pixels(x, num_samples_in_batch, node_coords)

        # Remove connection node embeddings
        h = h[np.where(node_type.detach().cpu().numpy() == 0)[0]]

        # Classify nodes for each channel seperately
        h_out = self.node_classifiers[0](h)
        for i in range(1, len(self.node_classifiers)):
            h_out = torch.cat([h_out, self.node_classifiers[i](h)], dim=1)

        return h_out.squeeze(1), None


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, output_size) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=(1, 1))
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=(1, 1))
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.AdaptiveMaxPool2d(output_size=output_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.BN1(x)
        x = F.relu(self.conv2(x))
        x = self.BN2(x)
        x = self.pool(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, output_size) -> None:
        super().__init__()
        self.upsample = nn.Upsample(size=output_size)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=(1, 1))
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=(1, 1))
        self.BN2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, x_skip):
        x = self.upsample(x) # F, N, N -> F, 2N, 2N
        x = F.relu(self.conv1(x))
        x = self.BN1(x) # F/2, 2N, 2N
        x = torch.cat([x, x_skip], dim=1) # F, 2N, 2N
        
        x = F.relu(self.conv2(x))
        x = self.BN2(x) # F, 2N, 2N
        return x


class IdenticalModel(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
    
    def forward(self, x):
        return x