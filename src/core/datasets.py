import numpy as np
import pandas as pd
import os
import bz2
import pickle
import ast
from abc import ABC
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx
import networkx as nx
import json
import glob
import logging
from pathlib import Path
import imageio
import math
import pandas as pd
import random
from torchvision.transforms.functional import hflip
import cv2


class UnityO:
    # This code is from https://data.unityimaging.net/upp.html

    # Like with DICOM - for the word image read instance.

    def __init__(self, unity_code: str, png_cache_dir=None, server_url=None, logger=None):
        self.png_cache_dir = png_cache_dir
        self.server_url = server_url

        unity_code = Path(unity_code)
        unity_code = unity_code.name.split('.')[0]
        unity_code = unity_code.replace("_(1)", "").replace("_(2)", "").replace("_(3)", "")

        self.unity_code = unity_code

        unity_code_len = len(self.unity_code)

        self._failed = False
        if unity_code_len == 67:
            self.code_type = 'video'
            self.unity_i_code = self.unity_code
        elif unity_code_len == 72:
            self.code_type = 'frame'
            self.frame_num = int(unity_code[-4:])
            self.unity_i_code = self.unity_code[:-5]
            self.unity_f_code = self.unity_code
        else:
            logger.warning(f"{unity_code} not a valid code")
            self._failed = True
            return

        self._sub_a = self.unity_i_code[:2]
        self._sub_b = self.unity_i_code[3:5]
        self._sub_c = self.unity_i_code[5:7]

    def get_frame_path(self, frame_offset=0):
        if self._failed:
            return ""

        if self.code_type == 'frame':
            return f"{self.png_cache_dir + '/' + self._sub_a + '/' + self._sub_b + '/' + self._sub_c + '/' + self.unity_i_code}-{(self.frame_num + frame_offset):04}.png"
        else:
            raise Exception

    def get_all_frames_path(self):
        search_string = f"{self.png_cache_dir / self._sub_a / self._sub_b / self._sub_c / self.unity_i_code}*.png"
        images_path = glob.glob(search_string)
        valid_images = sorted(images_path)

        return valid_images


class UICLVLandmark(Dataset, ABC):
    def __init__(self,
                 data_dir,
                 data_info_file,
                 mode,
                 num_aux_graphs,
                 logger=None,
                 transform=None,
                 frame_size=128,
                 average_coords=None,
                 main_graph_type='grid',
                 aux_graph_type='grid',
                 use_coordinate_graph=False,
                 use_connection_nodes=False,
                 use_main_graph_only=False,
                 image_crop_size=640,
                 image_out_size=608,
                 flip_p=0.0):

        super().__init__()

        if average_coords is None:
            # These numbers are obtained using the average_landmark_locations.py script
            self.average_coords = [[99.99, 112.57], [142.71, 90.67], [151.18, 86.25], [91.81, 117.91]]
        else:
            self.average_coords = average_coords

        self.deltas = pd.read_csv(os.path.join(data_info_file, '01_database_physical.csv'))

        # Read the labels file
        if mode == 'train':
            data_info_file = os.path.join(data_info_file, 'labels-train.json')
        elif mode == 'val':
            data_info_file = os.path.join(data_info_file, 'labels-tune.json')
        else:
            data_info_file = os.path.join(data_info_file, 'labels-test.json')

        with open(data_info_file, 'r') as json_file:
            self.data_info = json.load(json_file)

        self.unity_codes = list()
        for key in self.data_info.keys():
            hash = key.split('-')
            hash = hash[0] + '-' + hash[1]

            if len(self.deltas.loc[self.deltas['FileHash'] == hash, 'PhysicalDeltaX']) > 0 and \
                    self.all_coords_exist(self.data_info[key]['labels']):
                if not math.isnan(self.deltas.loc[self.deltas['FileHash'] == hash, 'PhysicalDeltaX'].iloc[0]):
                    self.unity_codes.append(key)

        # Create the graphs and the node types flag
        self.frame_size = frame_size
        self.num_aux_graphs = num_aux_graphs
        self.use_coordinate_graph = use_coordinate_graph
        self.use_connection_nodes = use_connection_nodes
        self.use_main_graph_only = use_main_graph_only

        self.graphs, self.node_type = self.create_graphs(main_graph_type, aux_graph_type)

        # Other required attributes
        self.mode = mode
        self.logger = logger
        self.transform = transform
        self.main_graph_type = main_graph_type
        self.data_dir = data_dir
        self.image_crop_size = image_crop_size
        self.image_out_size = image_out_size

    @staticmethod
    def all_coords_exist(label_dict):
        if label_dict['lv-ivs-top']['x'] and label_dict['lv-ivs-top']['y'] and \
                label_dict['lv-pw-top']['x'] and label_dict['lv-pw-top']['y'] and \
                label_dict['lv-ivs-bottom']['x'] and label_dict['lv-ivs-bottom']['y'] and \
                label_dict['lv-pw-bottom']['x'] and label_dict['lv-pw-bottom']['y']:
            return True
        else:
            return False

    @staticmethod
    def get_affine_matrix(tx=0, ty=0, sx=1, sy=1, rotation_theta=0, shear_theta=0):

        tf_rotate = torch.tensor([[math.cos(rotation_theta), -math.sin(rotation_theta), 0],
                                  [math.sin(rotation_theta), math.cos(rotation_theta), 0],
                                  [0, 0, 1]],
                                 dtype=torch.float)

        tf_translate = torch.tensor([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]],
                                    dtype=torch.float)

        tf_scale = torch.tensor([[sx, 0, 0],
                                 [0, sy, 0],
                                 [0, 0, 1]],
                                dtype=torch.float)

        tf_shear = torch.tensor([[1, -math.sin(shear_theta), 0],
                                 [0, math.cos(shear_theta), 0],
                                 [0, 0, 1]],
                                dtype=torch.float)

        matrix = tf_shear @ tf_scale @ tf_rotate @ tf_translate

        return matrix

    def get(self, idx):

        # # Get the data at index
        # data = self.data_info.iloc[idx]
        #
        # # Unpickle the data
        # pickle_file = bz2.BZ2File(data['cleaned_path'], 'rb')
        # mat_contents = pickle.load(pickle_file)
        # cine = mat_contents['resized'] # 224x224xN
        #
        # # Extracting the ED frame
        # if data['d_frame_number'] > cine.shape[-1]:
        #     ed_frame = cine[:, :, -1]
        # else:
        #     ed_frame = cine[:, :, data['d_frame_number']-1]

        # parse the unity code
        unity_o = UnityO(unity_code=self.unity_codes[idx], png_cache_dir=self.data_dir)
        frame_path = unity_o.get_frame_path()

        # Load the image
        frame, h_shift, w_shift, in_h, in_w = self.read_image_and_crop_into_tensor(image_path=frame_path,
                                                                       image_crop_size=(self.image_crop_size,
                                                                                        self.image_crop_size))

        in_out_ratio = self.image_crop_size / self.image_out_size

        transform_matrix = self.get_affine_matrix(tx=0,
                                                  ty=0,
                                                  sx=in_out_ratio,
                                                  sy=in_out_ratio,
                                                  rotation_theta=0,
                                                  shear_theta=0)

        transform_matrix_inv = transform_matrix.inverse()

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imsave('debug.png', frame.permute(1, 2, 0).detach().cpu().numpy())
        frame = frame.float().div(255)
        frame = self.transform_image(image=frame,
                                     transform_matrix=transform_matrix_inv,
                                     out_image_size=self.image_out_size)

        labels = self.data_info[self.unity_codes[idx]]['labels']
        xs = [float(labels[key]['x']) for key in ['lv-ivs-bottom', 'lv-pw-top', 'lv-pw-bottom', 'lv-ivs-top']]
        ys = [float(labels[key]['y']) for key in ['lv-ivs-bottom', 'lv-pw-top', 'lv-pw-bottom', 'lv-ivs-top']]
        coords = torch.tensor([ys, xs]).transpose(0, 1)
        label_shift = torch.tensor([h_shift, w_shift])
        coords = coords + label_shift

        coords = self.normalize_coord(coord=coords, image_size=self.image_crop_size)
        coords = self.apply_matrix_to_coords(transform_matrix=transform_matrix, coord=coords)
        coords = self.unnormalize_coord(coord=coords, image_size=self.image_out_size).squeeze().cpu().detach().numpy()
        coords = coords * self.frame_size / self.image_out_size
        coords = coords.astype('int')

        frame = self.transform(frame).unsqueeze(0)

        # frame[0, int(coord[0, 0, 0])-5:int(coord[0, 0, 0])+5, int(coord[0, 0, 1])-5:int(coord[0, 0, 1])+5] = 255
        # frame[0, int(coord[0, 1, 0])-5:int(coord[0, 1, 0])+5, int(coord[0, 1, 1])-5:int(coord[0, 1, 1])+5] = 255
        # frame[0, int(coord[0, 2, 0])-5:int(coord[0, 2, 0])+5, int(coord[0, 2, 1])-5:int(coord[0, 2, 1])+5] = 255
        # frame[0, int(coord[0, 3, 0])-5:int(coord[0, 3, 0])+5, int(coord[0, 3, 1])-5:int(coord[0, 3, 1])+5] = 255

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imsave('debugcrop.png', frame.permute(1, 2, 0).detach().cpu().numpy())

        # ed_frame shape = (224,224),  transform to torch tensor with shape (1,1,resized_size,resized_size)
        # orig_size = ed_frame.shape[0]
        # ed_frame = torch.tensor(ed_frame, dtype=torch.float32).unsqueeze(0) / 255  # (1, 224,224)
        # ed_frame = self.transform(ed_frame).unsqueeze(0)  # (1,1,frame_size,frame_size)

        # Extract landmark coordinates
        # coords = self.extract_coords(data, orig_size)

        # Create PyG data using the prebuilt networkx graph
        g = from_networkx(self.graphs)

        # Add the echo frame to pyG data
        g.x = frame

        # Create labels for each graph and add to PyG data
        g.y = torch.cat([self.create_node_labels(coord) for coord in coords], dim=1)
        g.valid_labels = torch.ones_like(g.y)

        # Add node type flag to PyG data
        g.node_type = torch.tensor(self.node_type)

        # Add initial location of nodes in the main graph and its labels
        if self.use_coordinate_graph and not self.use_main_graph_only:
            g.node_coords = torch.tensor(self.average_coords, dtype=torch.float32)
            g.node_coord_y = torch.tensor(coords, dtype=torch.float32)

        # Get the scale for each pixel in mm/pixel
        hash = self.unity_codes[idx].split('-')
        hash = hash[0]+'-'+hash[1]
        try:
            deltaX = float(self.deltas.loc[self.deltas['FileHash'] == hash, 'PhysicalDeltaX'].iloc[0])
            deltaY = float(self.deltas.loc[self.deltas['FileHash'] == hash, 'PhysicalDeltaY'].iloc[0])
        except IndexError:
            print("Delta not found for {}".format(hash))
            deltaY = 0.026
            deltaX = 0.026

        g.pix2mm_x = 10 * deltaX * in_w / self.frame_size # in mm
        g.pix2mm_y = 10 * deltaY * in_h / self.frame_size # in mm

        if math.isnan(g.pix2mm_x):
            g.pix2mm_x = 0.026*800/self.frame_size * 10
            g.pix2mm_y = 0.026*600/self.frame_size * 10
            print("Found NAN value!!!")

        g.pix2mm_x = torch.tensor(g.pix2mm_x, dtype=torch.float32)
        g.pix2mm_y = torch.tensor(g.pix2mm_y, dtype=torch.float32)

        return g

    @staticmethod
    def normalize_coord(coord: torch.Tensor, image_size: torch.Tensor):

        coord = (coord * 2 / image_size) - 1

        return coord

    @staticmethod
    def unnormalize_coord(coord: torch.Tensor, image_size: torch.tensor):

        coord = (coord + 1) * image_size / 2

        return coord

    def len(self):

        return len(self.unity_codes)

    def transform_image(self, image: torch.Tensor, transform_matrix: torch.Tensor, out_image_size=(608, 608)):

        device = image.device

        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)

        batch_size = image.shape[0]

        out_image_h = out_image_size
        out_image_w = out_image_size

        identity_grid = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32, device=device)
        intermediate_grid_shape = [batch_size, out_image_h * out_image_w, 2]

        grid = torch.nn.functional.affine_grid(identity_grid, [batch_size, 1, out_image_h, out_image_w],
                                               align_corners=False)
        grid = grid.reshape(intermediate_grid_shape)

        # For some reason it gives you w, h at the output of affine_grid. So switch here.
        grid = grid[..., [1, 0]]
        grid = self.apply_matrix_to_coords(transform_matrix=transform_matrix, coord=grid)
        grid = grid[..., [1, 0]]

        grid = grid.reshape([batch_size, out_image_h, out_image_w, 2])

        # There is no constant selection for padding mode - so border will have to do to weights.
        image = torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode="zeros",
                                                align_corners=False).squeeze(0)

        return image

    @ staticmethod
    def apply_matrix_to_coords(transform_matrix: torch.Tensor, coord: torch.Tensor):

        if coord.dim() == 2:
            coord = coord.unsqueeze(0)

        batch_size = coord.shape[0]

        if transform_matrix.dim() == 2:
            transform_matrix = transform_matrix.unsqueeze(0)

        if transform_matrix.size()[1:] == (3, 3):
            transform_matrix = transform_matrix[:, :2, :]

        A_batch = transform_matrix[:, :, :2]
        if A_batch.size(0) != batch_size:
            A_batch = A_batch.repeat(batch_size, 1, 1)

        B_batch = transform_matrix[:, :, 2].unsqueeze(1)

        coord = coord.bmm(A_batch.transpose(1, 2)) + B_batch.expand(coord.shape)

        return coord

    def create_graphs(self, main_graph_type, aux_graph_type):

        # List containing all graphs
        all_graphs = list()
        last_node = 0
        node_type = None
        self.inter_task_edges = list()

        if not self.use_main_graph_only:

            #  Graph connecting all aux graphs together
            if self.use_connection_nodes:
                connection_graph = nx.complete_graph(range(self.num_aux_graphs + 1))
                node_type = np.ones(connection_graph.number_of_nodes(), dtype=int) * 2
                all_graphs.append(connection_graph)
                last_node = connection_graph.number_of_nodes()

            for graph_num in range(1, self.num_aux_graphs + 1):

                # Number of patches along each dim
                patches_along_dim = 2 ** graph_num

                # Create a grid graph
                if aux_graph_type == 'grid' or aux_graph_type == 'grid-diagonal':
                    aux_graph = nx.grid_graph(dim=[range(last_node, last_node + patches_along_dim),
                                                   range(last_node, last_node + patches_along_dim)])

                # Add the diagonal edges for grid-diagonal graphs
                if aux_graph_type == 'grid-diagonal':
                    # Code from https://stackoverflow.com/questions/55772715/how-to-create-8-cell-adjacency-map-for-a-diagonal-enabled-a-algorithm-with-the
                    edges = [((x, y), (x + 1, y + 1)) for x in range(last_node, last_node + patches_along_dim - 1) for y
                             in
                             range(last_node, last_node + patches_along_dim - 1)] + \
                            [((x + 1, y), (x, y + 1)) for x in range(last_node, last_node + patches_along_dim - 1) for y
                             in
                             range(last_node, last_node + patches_along_dim - 1)]
                    aux_graph.add_edges_from(edges)

                # Update flag indicating which nodes are connection (virtual) nodes
                node_type = np.hstack((node_type, np.zeros(aux_graph.number_of_nodes()))) if node_type is not None \
                    else np.zeros(aux_graph.number_of_nodes())

                # Add graphs together into a single graph
                all_graphs.append(aux_graph)
                last_node = list(all_graphs[-1].nodes)[-1][-1] + 1

            for graph_num in range(1, self.num_aux_graphs):
                self.add_inter_aux_task_edges(all_graphs, graph_num)

        # Create main grid graph
        if main_graph_type == 'grid' or main_graph_type == 'grid-diagonal':
            main_graph = nx.grid_graph(dim=[range(last_node, last_node + self.frame_size),
                                            range(last_node, last_node + self.frame_size)])

        # Add the diagonal edges for the grid-diagonal main graph
        if main_graph_type == 'grid-diagonal':
            # Code from https://stackoverflow.com/questions/55772715/how-to-create-8-cell-adjacency-map-for-a-diagonal-enabled-a-algorithm-with-the
            edges = [((x, y), (x + 1, y + 1)) for x in range(last_node, last_node + self.frame_size - 1) for y in
                     range(last_node, last_node + self.frame_size - 1)] + \
                    [((x + 1, y), (x, y + 1)) for x in range(last_node, last_node + self.frame_size - 1) for y in
                     range(last_node, last_node + self.frame_size - 1)]
            main_graph.add_edges_from(edges)

        # Add main graph to list of graphs
        all_graphs.append(main_graph)
        last_node = list(all_graphs[-1].nodes)[-1][-1] + 1
        node_type = np.hstack((node_type, np.zeros(main_graph.number_of_nodes()))) if node_type is not None \
            else np.zeros(main_graph.number_of_nodes())

        # Add edges between the finest aux graph and the main graph
        if not self.use_main_graph_only:
            self.add_inter_main_task_edges(all_graphs, self.num_aux_graphs)

            if self.use_connection_nodes:
                for graph_num in range(1, self.num_aux_graphs):
                    nodes = list(all_graphs[graph_num])
                    self.inter_task_edges = self.inter_task_edges + [(graph_num - 1, nodes[i]) for i in
                                                                     range(len(nodes))]

            # Create the coordinate graph (with only 4 nodes) for the main task
            if self.use_coordinate_graph:
                coord_graph = nx.complete_graph(range(last_node, last_node + 4))

                node_type = np.hstack((node_type, np.ones(coord_graph.number_of_nodes())))

                all_graphs.append(coord_graph)

        # Consolidate all graphs
        for i, graph in enumerate(all_graphs):
            graphs = nx.compose(graphs, graph) if i != 0 else graph

        # Add the additional edges to graphs
        graphs.add_edges_from(self.inter_task_edges)

        return graphs, node_type

    def add_inter_aux_task_edges(self, all_graphs, graph_num):

        if self.use_connection_nodes:
            all_graphs = all_graphs[1:]

        source_nodes = np.reshape(np.array(all_graphs[graph_num - 1].nodes), (2 ** graph_num, 2 ** graph_num, 2))
        destination_nodes = np.reshape(np.array(all_graphs[graph_num].nodes), (2 ** (graph_num + 1),
                                                                               2 ** (graph_num + 1), 2))

        for x in range(source_nodes.shape[0]):
            for y in range(source_nodes.shape[1]):
                initial_dest_x = x * 2
                initial_dest_y = y * 2

                last_dest_x = initial_dest_x + 2
                last_dest_y = initial_dest_y + 2

                dst_nodes = np.reshape(destination_nodes[initial_dest_x: last_dest_x, initial_dest_y: last_dest_y,
                                       :],
                                       (4, 2))

                self.inter_task_edges = self.inter_task_edges + [(tuple(source_nodes[x, y, :]), tuple(dst_node))
                                                                 for dst_node in dst_nodes]

    def add_inter_main_task_edges(self, all_graphs, graph_num):

        if self.use_connection_nodes:
            all_graphs = all_graphs[1:]

        source_nodes = np.reshape(np.array(all_graphs[graph_num - 1].nodes), (2 ** graph_num, 2 ** graph_num, 2))

        center_loc = (source_nodes.shape[0] - self.frame_size // 2) // 2
        source_nodes = source_nodes[center_loc: center_loc + self.frame_size // 2,
                       center_loc: center_loc + self.frame_size // 2, :]
        destination_nodes = np.reshape(np.array(all_graphs[graph_num].nodes), (self.frame_size,
                                                                               self.frame_size, 2))

        for x in range(source_nodes.shape[0]):
            for y in range(source_nodes.shape[1]):
                initial_dest_x = x * 2
                initial_dest_y = y * 2

                last_dest_x = initial_dest_x + 2
                last_dest_y = initial_dest_y + 2

                dst_nodes = np.reshape(destination_nodes[initial_dest_x: last_dest_x, initial_dest_y: last_dest_y,
                                       :],
                                       (4, 2))

                self.inter_task_edges = self.inter_task_edges + [(tuple(source_nodes[x, y, :]), tuple(dst_node))
                                                                 for dst_node in dst_nodes]

    def create_node_labels(self, coordinates):

        # Only a single coordinate is to be passed to this func

        y = None

        # Add the labels for the aux graphs
        if not self.use_main_graph_only:
            for graph_num in range(1, self.num_aux_graphs + 1):
                bins = np.linspace(start=0, stop=self.frame_size, num=2 ** graph_num + 1)
                transformed_coordinates = np.digitize(coordinates, bins=bins) - 1
                transformed_coordinates = tuple(np.array(transformed_coordinates).T)

                y_temp = np.zeros((2 ** graph_num, 2 ** graph_num))
                y_temp[transformed_coordinates] = 1.0

                y = torch.cat([y, torch.flatten(torch.tensor(y_temp, dtype=torch.float32))], dim=0) \
                    if graph_num != 1 else torch.flatten(torch.tensor(y_temp, dtype=torch.float32))

        # Only add labels for the main graph if it's not using the 4-node coordinate implementation
        transformed_coordinates = tuple(np.array(coordinates).T)
        y_temp = np.zeros((self.frame_size, self.frame_size))
        y_temp[transformed_coordinates] = 1.0
        y = torch.cat([y, torch.flatten(torch.tensor(y_temp, dtype=torch.float32))], dim=0) if y is not None else \
            torch.flatten(torch.tensor(y_temp, dtype=torch.float32))

        return y.view((-1, 1))

    @staticmethod
    def read_image_and_crop_into_tensor(image_path, image_crop_size=(640, 640)):
        # This code is from https://data.unityimaging.net/upp.html

        out_h = image_crop_size[0]
        out_w = image_crop_size[1]

        image = torch.zeros((3, out_h, out_w), dtype=torch.uint8)

        try:
            image_np = imageio.imread(image_path)
            if image_np.ndim == 2:
                image_np = np.stack((image_np, image_np, image_np), axis=-1)
        except Exception as e:
            print(f"Failed to load image: {image_path}")
            return image, 0, 0

        in_h = image_np.shape[0]
        in_w = image_np.shape[1]

        if in_h <= out_h:
            in_s_h = 0
            in_e_h = in_s_h + in_h
            out_s_h = (out_h - in_h) // 2
            out_e_h = out_s_h + in_h
            label_height_shift = out_s_h
        else:
            in_s_h = (in_h - out_h) // 2
            in_e_h = in_s_h + out_h
            out_s_h = 0
            out_e_h = out_s_h + out_h
            label_height_shift = -in_s_h

        if in_w <= out_w:
            in_s_w = 0
            in_e_w = in_s_w + in_w
            out_s_w = (out_w - in_w) // 2
            out_e_w = out_s_w + in_w
            label_width_shift = out_s_w
        else:
            in_s_w = (in_w - out_w) // 2
            in_e_w = in_s_w + out_w
            out_s_w = 0
            out_e_w = out_s_w + out_w
            label_width_shift = -in_s_w

        image[:, out_s_h:out_e_h, out_s_w:out_e_w] = torch.tensor(image_np[in_s_h:in_e_h, in_s_w:in_e_w, :]).permute(2, 0, 1)

        return image, label_height_shift, label_width_shift, in_h, in_w


class LVLandmark(Dataset, ABC):
    def __init__(self,
                 data_dir,
                 data_info_file,
                 mode,
                 num_aux_graphs,
                 logger=None,
                 transform=None,
                 frame_size=128,
                 average_coords=None,
                 main_graph_type='grid',
                 aux_graph_type='grid',
                 use_coordinate_graph=False,
                 use_connection_nodes=False,
                 use_main_graph_only=False,
                 flip_p=0.0):

        super().__init__()

        if average_coords is None:
            # These numbers are obtained using the average_landmark_locations.py script
            self.average_coords = [[99.99, 112.57], [142.71, 90.67], [151.18, 86.25], [91.81, 117.91]]
        else:
            self.average_coords = average_coords

        # Read the data CSV file
        self.data_info = pd.read_csv(data_info_file)

        # Rename the index column so the processed data can be tracked down later
        self.data_info = self.data_info.rename(columns={'Unnamed: 0': 'db_indx'})

        # Get the correct data split
        self.data_info = self.data_info[self.data_info.split == mode]

        if logger is not None:
            logger.info(f'#{mode}: {len(self.data_info)}')

        # Add root directory to file names to create full path to data
        self.data_info['cleaned_path'] = self.data_info.apply(lambda row: os.path.join(data_dir, row['file_name']),
                                                              axis=1)

        # Create the graphs and the node types flag
        self.frame_size = frame_size
        self.num_aux_graphs = num_aux_graphs
        self.use_coordinate_graph = use_coordinate_graph
        self.use_connection_nodes = use_connection_nodes
        self.use_main_graph_only = use_main_graph_only

        self.graphs, self.node_type = self.create_graphs(main_graph_type, aux_graph_type)

        # Other required attributes
        self.mode = mode
        self.logger = logger
        self.transform = transform
        self.main_graph_type = main_graph_type
        self.flip_p = flip_p

    def __getitem__(self, idx):

        # Get the data at index
        data = self.data_info.iloc[idx]

        # Unpickle the data
        pickle_file = bz2.BZ2File(data['cleaned_path'], 'rb')
        mat_contents = pickle.load(pickle_file)
        cine = mat_contents['resized'] # 224x224xN

        # Extracting the ED frame
        if data['d_frame_number'] > cine.shape[-1]:
            ed_frame = cine[:, :, -1]
        else:
            ed_frame = cine[:, :, data['d_frame_number']-1]

        # ed_frame shape = (224,224),  transform to torch tensor with shape (1,1,resized_size,resized_size)
        orig_size = ed_frame.shape[0]
        ed_frame = torch.tensor(ed_frame, dtype=torch.float32).unsqueeze(0) / 255  # (1, 224,224)
        ed_frame = self.transform(ed_frame).unsqueeze(0)  # (1,1,frame_size,frame_size)

        # Extract landmark coordinates
        coords = self.extract_coords(data, orig_size)

        if random.uniform(0, 1) <= self.flip_p and self.mode == "train":
            coords[:, 1] = self.frame_size - coords[:, 1] - 1
            ed_frame = hflip(ed_frame)

        # Create PyG data using the prebuilt networkx graph
        g = from_networkx(self.graphs)

        # Add the echo frame to pyG data
        g.x = ed_frame

        # Create labels for each graph and add to PyG data
        g.y = torch.cat([self.create_node_labels(coord) for coord in coords], dim=1)
        g.valid_labels = torch.ones_like(g.y)

        # Add node type flag to PyG data
        g.node_type = torch.tensor(self.node_type)

        # Add initial location of nodes in the main graph and its labels
        if self.use_coordinate_graph and not self.use_main_graph_only:
            g.node_coords = torch.tensor(self.average_coords, dtype=torch.float32)
            g.node_coord_y = torch.tensor(coords, dtype=torch.float32)

        # Get the scale for each pixel in mm/pixel
        deltaX = data['DeltaX']*orig_size/self.frame_size
        deltaY = data['DeltaY']*orig_size/self.frame_size
        g.pix2mm_x = torch.tensor(deltaX * 10, dtype=torch.float32)  # in mm
        g.pix2mm_y = torch.tensor(deltaY * 10, dtype=torch.float32)  # in mm

        # Adding other header data to PyG data
        if self.mode != 'train':  # collecting header data for evaluation section
            keys = ['db_indx', "PatientID", "StudyDate", "SIUID", "LV_Mass", "BSA", "file_name"]
            g.update(data[keys].to_dict())

        return g

    def __len__(self):

        return len(self.data_info)

    def extract_coords(self, df, orig_frame_size):

        # get all landmark coordinates, select the four we need
        LVIDd_coordinate = np.round(np.array(ast.literal_eval(df['LVID'])) * self.frame_size / orig_frame_size).astype(int)
        IVS_coordinates = np.round(np.array(ast.literal_eval(df['IVS'])) * self.frame_size / orig_frame_size).astype(int)
        LVPW_coordinates = np.round(np.array(ast.literal_eval(df['LVPW'])) * self.frame_size / orig_frame_size).astype(int)

        # Note that the coordinates are saved in (h, w) convention. in order: LVID_top, LVID_bot, LVPW, IVS
        coords = []
        coords.append([LVIDd_coordinate[1] - 1, LVIDd_coordinate[0] - 1])
        coords.append([LVIDd_coordinate[3] - 1, LVIDd_coordinate[2] - 1])
        coords.append([LVPW_coordinates[3] - 1, LVPW_coordinates[2] - 1])
        coords.append([IVS_coordinates[1] - 1, IVS_coordinates[0] - 1])
        coords = np.array(coords)

        return coords

    def create_graphs(self, main_graph_type, aux_graph_type):

        # List containing all graphs
        all_graphs = list()
        last_node = 0
        node_type = None
        self.inter_task_edges = list()

        if not self.use_main_graph_only:

            #  Graph connecting all aux graphs together
            if self.use_connection_nodes:
                connection_graph = nx.complete_graph(range(self.num_aux_graphs + 1))
                node_type = np.ones(connection_graph.number_of_nodes(), dtype=int) * 2
                all_graphs.append(connection_graph)
                last_node = connection_graph.number_of_nodes()

            for graph_num in range(1, self.num_aux_graphs + 1):

                # Number of patches along each dim
                patches_along_dim = 2 ** graph_num

                # Create a grid graph
                if aux_graph_type == 'grid' or aux_graph_type == 'grid-diagonal':
                    aux_graph = nx.grid_graph(dim=[range(last_node, last_node + patches_along_dim),
                                                   range(last_node, last_node + patches_along_dim)])

                # Add the diagonal edges for grid-diagonal graphs
                if aux_graph_type == 'grid-diagonal':
                    # Code from https://stackoverflow.com/questions/55772715/how-to-create-8-cell-adjacency-map-for-a-diagonal-enabled-a-algorithm-with-the
                    edges = [((x, y), (x + 1, y + 1)) for x in range(last_node, last_node + patches_along_dim - 1) for y
                             in
                             range(last_node, last_node + patches_along_dim - 1)] + \
                            [((x + 1, y), (x, y + 1)) for x in range(last_node, last_node + patches_along_dim - 1) for y
                             in
                             range(last_node, last_node + patches_along_dim - 1)]
                    aux_graph.add_edges_from(edges)

                # Update flag indicating which nodes are connection (virtual) nodes
                node_type = np.hstack((node_type, np.zeros(aux_graph.number_of_nodes()))) if node_type is not None \
                    else np.zeros(aux_graph.number_of_nodes())

                # Add graphs together into a single graph
                all_graphs.append(aux_graph)
                last_node = list(all_graphs[-1].nodes)[-1][-1] + 1

            for graph_num in range(1, self.num_aux_graphs):
                self.add_inter_aux_task_edges(all_graphs, graph_num)

        # Create main grid graph
        if main_graph_type == 'grid' or main_graph_type == 'grid-diagonal':
            main_graph = nx.grid_graph(dim=[range(last_node, last_node + self.frame_size),
                                            range(last_node, last_node + self.frame_size)])

        # Add the diagonal edges for the grid-diagonal main graph
        if main_graph_type == 'grid-diagonal':
            # Code from https://stackoverflow.com/questions/55772715/how-to-create-8-cell-adjacency-map-for-a-diagonal-enabled-a-algorithm-with-the
            edges = [((x, y), (x + 1, y + 1)) for x in range(last_node, last_node + self.frame_size - 1) for y in
                     range(last_node, last_node + self.frame_size - 1)] + \
                    [((x + 1, y), (x, y + 1)) for x in range(last_node, last_node + self.frame_size - 1) for y in
                     range(last_node, last_node + self.frame_size - 1)]
            main_graph.add_edges_from(edges)

        # Add main graph to list of graphs
        all_graphs.append(main_graph)
        last_node = list(all_graphs[-1].nodes)[-1][-1] + 1
        node_type = np.hstack((node_type, np.zeros(main_graph.number_of_nodes()))) if node_type is not None \
            else np.zeros(main_graph.number_of_nodes())

        # Add edges between the finest aux graph and the main graph
        if not self.use_main_graph_only:
            self.add_inter_main_task_edges(all_graphs, self.num_aux_graphs)

            if self.use_connection_nodes:
                for graph_num in range(1, self.num_aux_graphs):
                    nodes = list(all_graphs[graph_num])
                    self.inter_task_edges = self.inter_task_edges + [(graph_num - 1, nodes[i]) for i in
                                                                     range(len(nodes))]

            # Create the coordinate graph (with only 4 nodes) for the main task
            if self.use_coordinate_graph:
                coord_graph = nx.complete_graph(range(last_node, last_node + 4))

                node_type = np.hstack((node_type, np.ones(coord_graph.number_of_nodes())))

                all_graphs.append(coord_graph)

        # Consolidate all graphs
        for i, graph in enumerate(all_graphs):
            graphs = nx.compose(graphs, graph) if i != 0 else graph

        # Add the additional edges to graphs
        graphs.add_edges_from(self.inter_task_edges)

        return graphs, node_type

    def add_inter_aux_task_edges(self, all_graphs, graph_num):

        if self.use_connection_nodes:
            all_graphs = all_graphs[1:]

        source_nodes = np.reshape(np.array(all_graphs[graph_num - 1].nodes), (2 ** graph_num, 2 ** graph_num, 2))
        destination_nodes = np.reshape(np.array(all_graphs[graph_num].nodes), (2 ** (graph_num + 1),
                                                                               2 ** (graph_num + 1), 2))

        for x in range(source_nodes.shape[0]):
            for y in range(source_nodes.shape[1]):
                initial_dest_x = x * 2
                initial_dest_y = y * 2

                last_dest_x = initial_dest_x + 2
                last_dest_y = initial_dest_y + 2

                dst_nodes = np.reshape(destination_nodes[initial_dest_x: last_dest_x, initial_dest_y: last_dest_y,
                                       :],
                                       (4, 2))

                self.inter_task_edges = self.inter_task_edges + [(tuple(source_nodes[x, y, :]), tuple(dst_node))
                                                                 for dst_node in dst_nodes]

    def add_inter_main_task_edges(self, all_graphs, graph_num):

        if self.use_connection_nodes:
            all_graphs = all_graphs[1:]

        source_nodes = np.reshape(np.array(all_graphs[graph_num - 1].nodes), (2 ** graph_num, 2 ** graph_num, 2))

        center_loc = (source_nodes.shape[0] - self.frame_size // 2) // 2
        source_nodes = source_nodes[center_loc: center_loc + self.frame_size // 2,
                       center_loc: center_loc + self.frame_size // 2, :]
        destination_nodes = np.reshape(np.array(all_graphs[graph_num].nodes), (self.frame_size,
                                                                               self.frame_size, 2))

        for x in range(source_nodes.shape[0]):
            for y in range(source_nodes.shape[1]):
                initial_dest_x = x * 2
                initial_dest_y = y * 2

                last_dest_x = initial_dest_x + 2
                last_dest_y = initial_dest_y + 2

                dst_nodes = np.reshape(destination_nodes[initial_dest_x: last_dest_x, initial_dest_y: last_dest_y,
                                       :],
                                       (4, 2))

                self.inter_task_edges = self.inter_task_edges + [(tuple(source_nodes[x, y, :]), tuple(dst_node))
                                                                 for dst_node in dst_nodes]

    def create_node_labels(self, coordinates):

        # Only a single coordinate is to be passed to this func

        y = None

        # Add the labels for the aux graphs
        if not self.use_main_graph_only:
            for graph_num in range(1, self.num_aux_graphs + 1):
                bins = np.linspace(start=0, stop=self.frame_size, num=2 ** graph_num + 1)
                transformed_coordinates = np.digitize(coordinates, bins=bins) - 1
                transformed_coordinates = tuple(np.array(transformed_coordinates).T)

                y_temp = np.zeros((2 ** graph_num, 2 ** graph_num))
                y_temp[transformed_coordinates] = 1.0

                y = torch.cat([y, torch.flatten(torch.tensor(y_temp, dtype=torch.float32))], dim=0) \
                    if graph_num != 1 else torch.flatten(torch.tensor(y_temp, dtype=torch.float32))

        # Only add labels for the main graph if it's not using the 4-node coordinate implementation
        transformed_coordinates = tuple(np.array(coordinates).T)
        y_temp = np.zeros((self.frame_size, self.frame_size))
        y_temp[transformed_coordinates] = 1.0
        y = torch.cat([y, torch.flatten(torch.tensor(y_temp, dtype=torch.float32))], dim=0) if y is not None else \
            torch.flatten(torch.tensor(y_temp, dtype=torch.float32))

        return y.view((-1, 1))


class EchoNetLandmark(Dataset, ABC):
    def __init__(self,
                 data_dir,
                 data_info_file,
                 mode,
                 num_aux_graphs,
                 logger=None,
                 transform=None,
                 frame_size=128,
                 average_coords=None,
                 main_graph_type='grid',
                 aux_graph_type='grid',
                 use_coordinate_graph=False,
                 use_connection_nodes=False,
                 use_main_graph_only=False,
                 flip_p=0.0):

        super().__init__()

        self.frame_size = frame_size

        if average_coords is None:
            # These numbers are obtained using the average_landmark_locations.py script
            self.average_coords = [[99.99, 112.57], [142.71, 90.67], [151.18, 86.25], [91.81, 117.91]]
        else:
            self.average_coords = average_coords

        # Read the data CSV file
        self.data_info = pd.read_csv(data_info_file)

        # Get the correct data split
        self.data_info = self.data_info[self.data_info.split == mode]
        # self.data_info = self.data_info.head(10)

        # Create the list containing the path to each video
        vid_dirs = set(self.data_info.apply(lambda row: os.path.join(data_dir, row['HashedFileName']),
                                                              axis=1))

        self.data_list = list()
        for vid_dir in vid_dirs:
            vid_rows = self.data_info.loc[self.data_info['HashedFileName'] == os.path.basename(vid_dir)]
            frame_nums = set(vid_rows['Frame'])
            resolution = list(vid_rows['pix2mm'])[0]

            for frame_num in frame_nums:
                self.data_list.append({'vid_dir': vid_dir, 'frame_num': frame_num, 'resolution': resolution})

                lvid, ivs, lvpw = self._extract_coords_from_df(vid_rows, frame_num)

                self.data_list[-1].update({'coords': {'lvid': lvid,
                                                      'ivs': ivs,
                                                      'lvpw': lvpw}})

        if logger is not None:
            logger.info(f'#{mode}: {len(self.data_list)}')

        # Create the graphs and the node types flag
        self.num_aux_graphs = num_aux_graphs
        self.use_coordinate_graph = use_coordinate_graph
        self.use_connection_nodes = use_connection_nodes
        self.use_main_graph_only = use_main_graph_only

        self.graphs, self.node_type = self.create_graphs(main_graph_type, aux_graph_type)

        # Other required attributes
        self.mode = mode
        self.logger = logger
        self.transform = transform
        self.main_graph_type = main_graph_type
        self.flip_p = flip_p

    def _extract_coords_from_df(self, df, frame_num):

        lvidd = df.loc[(df['Frame'] == frame_num) & (df['Calc'] == 'LVIDd')]
        lvids = df.loc[(df['Frame'] == frame_num) & (df['Calc'] == 'LVIDs')]
        orig_h = df.loc[(df['Frame'] == frame_num)]['Height'].iloc[0]
        orig_w = df.loc[(df['Frame'] == frame_num)]['Width'].iloc[0]
        lvid = pd.concat([lvidd, lvids], ignore_index=True)
        lvid = [[lvid['Y1'].iloc[0] * self.frame_size/orig_h,
                lvid['X1'].iloc[0] * self.frame_size/orig_w],
                [lvid['Y2'].iloc[0] * self.frame_size/orig_h,
                lvid['X2'].iloc[0] * self.frame_size/orig_w]] if len(lvid) > 0 else None

        ivsd = df.loc[(df['Frame'] == frame_num) & (df['Calc'] == 'IVSd')]
        ivss = df.loc[(df['Frame'] == frame_num) & (df['Calc'] == 'IVSs')]
        ivs = pd.concat([ivsd, ivss])
        ivs = [[ivs['Y1'].iloc[0]* self.frame_size/orig_h,
               ivs['X1'].iloc[0]* self.frame_size/orig_w],
               [ivs['Y2'].iloc[0]* self.frame_size/orig_h,
               ivs['X2'].iloc[0]* self.frame_size/orig_w]] if len(ivs) > 0 else None

        lvpwd = df.loc[(df['Frame'] == frame_num) & (df['Calc'] == 'LVPWd')]
        lvpws = df.loc[(df['Frame'] == frame_num) & (df['Calc'] == 'LVPWs')]
        lvpw = pd.concat([lvpwd, lvpws])
        lvpw = [[lvpw['Y1'].iloc[0]* self.frame_size/orig_h,
                lvpw['X1'].iloc[0]* self.frame_size/orig_w],
                [lvpw['Y2'].iloc[0]* self.frame_size/orig_h,
                lvpw['X2'].iloc[0]* self.frame_size/orig_w]] if len(lvpw) > 0 else None

        # Round coords
        lvid = lvid if lvid is None else np.round(np.array(lvid))
        ivs = ivs if ivs is None else np.round(np.array(ivs))
        lvpw = lvpw if lvpw is None else np.round(np.array(lvpw))

        return lvid, ivs, lvpw

    def __getitem__(self, idx):

        # Get the data dict
        data = self.data_list[idx]
        vid_dir = data["vid_dir"]
        frame_num = data["frame_num"]

        # Load the video and get the right frame
        ed_frame = self._loadvideo(vid_dir+'.avi')
        ed_frame = ed_frame[:, :, frame_num]

        # ed_frame shape = (224,224),  transform to torch tensor with shape (1,1,resized_size,resized_size)
        orig_size = ed_frame.shape[0]
        ed_frame = torch.tensor(ed_frame, dtype=torch.float32).unsqueeze(0) / 255  # (1, 224,224)
        ed_frame = self.transform(ed_frame).unsqueeze(0)  # (1,1,frame_size,frame_size)

        # Extract landmark coordinates
        coords = data['coords']
        coords = self.process_coords(coords)

        if random.uniform(0, 1) <= self.flip_p and self.mode == "train":
            coords[:, 1] = self.frame_size - coords[:, 1] - 1
            ed_frame = hflip(ed_frame)

        # Create PyG data using the prebuilt networkx graph
        g = from_networkx(self.graphs)

        # Add the echo frame to pyG data
        g.x = ed_frame

        # Create labels for each graph and add to PyG data
        node_labels, valid_labels = self.create_node_labels(coords, (self.node_type[self.node_type == 0].shape[0], 1), idx)

        g.y = node_labels
        g.valid_labels = valid_labels

        # Add node type flag to PyG data
        g.node_type = torch.tensor(self.node_type)

        # Add initial location of nodes in the main graph and its labels
        if self.use_coordinate_graph and not self.use_main_graph_only:
            g.node_coords = torch.tensor(self.average_coords, dtype=torch.float32)
            g.node_coord_y = torch.tensor(coords, dtype=torch.float32)

        # Get the scale for each pixel in mm/pixel
        g.pix2mm_x = torch.tensor(data['resolution'], dtype=torch.float32)  # in mm
        g.pix2mm_y = torch.tensor(data['resolution'], dtype=torch.float32)  # in mm

        return g

    def __len__(self):

        return len(self.data_list)

    def process_coords(self, coords_dict):
        coords = np.zeros((4, 2))

        # Find the first coordinate (Bottom of IVS or TOP of LVID)
        if coords_dict['ivs'] is not None and coords_dict['lvid'] is not None:
            coords[0] = (coords_dict['ivs'][0] + coords_dict['lvid'][1]) // 2
        elif coords_dict['ivs'] is not None:
            coords[0] = coords_dict['ivs'][0]
        elif coords_dict['lvid'] is not None:
            coords[0] = coords_dict['lvid'][1]
        else:
            coords[0] = [-1, -1]

        # Find the second coordinate (Top of LVPW or Bottomw of LVID)
        if coords_dict['lvpw'] is not None and coords_dict['lvid'] is not None:
            coords[1] = (coords_dict['lvpw'][1] + coords_dict['lvid'][0]) // 2
        elif coords_dict['lvpw'] is not None:
            coords[1] = coords_dict['lvpw'][1]
        elif coords_dict['lvid'] is not None:
            coords[1] = coords_dict['lvid'][0]
        else:
            coords[1] = [-1, -1]

        # Find fourth coordinate (Bottom of LVPW)
        if coords_dict['lvpw'] is not None:
            coords[2] = coords_dict['lvpw'][0]
        else:
            coords[2] = [-1, -1]

        # Find the third coordinate (Top of IVS)
        if coords_dict['ivs'] is not None:
            coords[3] = coords_dict['ivs'][1]
        else:
            coords[3] = [-1, -1]

        return coords

    @staticmethod
    def _loadvideo(filename: str):
        """
        Video loader code from https://github.com/echonet/dynamic/tree/master/echonet with some modifications

        :param filename: str, path to video to load
        :return: numpy array of dimension H*W*T
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v = np.zeros((frame_height, frame_width, frame_count), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            v[:, :, count] = frame

        return v

    def create_graphs(self, main_graph_type, aux_graph_type):

        # List containing all graphs
        all_graphs = list()
        last_node = 0
        node_type = None
        self.inter_task_edges = list()

        if not self.use_main_graph_only:

            #  Graph connecting all aux graphs together
            if self.use_connection_nodes:
                connection_graph = nx.complete_graph(range(self.num_aux_graphs + 1))
                node_type = np.ones(connection_graph.number_of_nodes(), dtype=int) * 2
                all_graphs.append(connection_graph)
                last_node = connection_graph.number_of_nodes()

            for graph_num in range(1, self.num_aux_graphs + 1):

                # Number of patches along each dim
                patches_along_dim = 2 ** graph_num

                # Create a grid graph
                if aux_graph_type == 'grid' or aux_graph_type == 'grid-diagonal':
                    aux_graph = nx.grid_graph(dim=[range(last_node, last_node + patches_along_dim),
                                                   range(last_node, last_node + patches_along_dim)])

                # Add the diagonal edges for grid-diagonal graphs
                if aux_graph_type == 'grid-diagonal':
                    # Code from https://stackoverflow.com/questions/55772715/how-to-create-8-cell-adjacency-map-for-a-diagonal-enabled-a-algorithm-with-the
                    edges = [((x, y), (x + 1, y + 1)) for x in range(last_node, last_node + patches_along_dim - 1) for y
                             in
                             range(last_node, last_node + patches_along_dim - 1)] + \
                            [((x + 1, y), (x, y + 1)) for x in range(last_node, last_node + patches_along_dim - 1) for y
                             in
                             range(last_node, last_node + patches_along_dim - 1)]
                    aux_graph.add_edges_from(edges)

                # Update flag indicating which nodes are connection (virtual) nodes
                node_type = np.hstack((node_type, np.zeros(aux_graph.number_of_nodes()))) if node_type is not None \
                    else np.zeros(aux_graph.number_of_nodes())

                # Add graphs together into a single graph
                all_graphs.append(aux_graph)
                last_node = list(all_graphs[-1].nodes)[-1][-1] + 1

            for graph_num in range(1, self.num_aux_graphs):
                self.add_inter_aux_task_edges(all_graphs, graph_num)

        # Create main grid graph
        if main_graph_type == 'grid' or main_graph_type == 'grid-diagonal':
            main_graph = nx.grid_graph(dim=[range(last_node, last_node + self.frame_size),
                                            range(last_node, last_node + self.frame_size)])

        # Add the diagonal edges for the grid-diagonal main graph
        if main_graph_type == 'grid-diagonal':
            # Code from https://stackoverflow.com/questions/55772715/how-to-create-8-cell-adjacency-map-for-a-diagonal-enabled-a-algorithm-with-the
            edges = [((x, y), (x + 1, y + 1)) for x in range(last_node, last_node + self.frame_size - 1) for y in
                     range(last_node, last_node + self.frame_size - 1)] + \
                    [((x + 1, y), (x, y + 1)) for x in range(last_node, last_node + self.frame_size - 1) for y in
                     range(last_node, last_node + self.frame_size - 1)]
            main_graph.add_edges_from(edges)

        # Add main graph to list of graphs
        all_graphs.append(main_graph)
        last_node = list(all_graphs[-1].nodes)[-1][-1] + 1
        node_type = np.hstack((node_type, np.zeros(main_graph.number_of_nodes()))) if node_type is not None \
            else np.zeros(main_graph.number_of_nodes())

        # Add edges between the finest aux graph and the main graph
        if not self.use_main_graph_only:
            self.add_inter_main_task_edges(all_graphs, self.num_aux_graphs)

            if self.use_connection_nodes:
                for graph_num in range(1, self.num_aux_graphs):
                    nodes = list(all_graphs[graph_num])
                    self.inter_task_edges = self.inter_task_edges + [(graph_num - 1, nodes[i]) for i in
                                                                     range(len(nodes))]

            # Create the coordinate graph (with only 4 nodes) for the main task
            if self.use_coordinate_graph:
                coord_graph = nx.complete_graph(range(last_node, last_node + 4))

                node_type = np.hstack((node_type, np.ones(coord_graph.number_of_nodes())))

                all_graphs.append(coord_graph)

        # Consolidate all graphs
        for i, graph in enumerate(all_graphs):
            graphs = nx.compose(graphs, graph) if i != 0 else graph

        # Add the additional edges to graphs
        graphs.add_edges_from(self.inter_task_edges)

        return graphs, node_type

    def add_inter_aux_task_edges(self, all_graphs, graph_num):

        if self.use_connection_nodes:
            all_graphs = all_graphs[1:]

        source_nodes = np.reshape(np.array(all_graphs[graph_num - 1].nodes), (2 ** graph_num, 2 ** graph_num, 2))
        destination_nodes = np.reshape(np.array(all_graphs[graph_num].nodes), (2 ** (graph_num + 1),
                                                                               2 ** (graph_num + 1), 2))

        for x in range(source_nodes.shape[0]):
            for y in range(source_nodes.shape[1]):
                initial_dest_x = x * 2
                initial_dest_y = y * 2

                last_dest_x = initial_dest_x + 2
                last_dest_y = initial_dest_y + 2

                dst_nodes = np.reshape(destination_nodes[initial_dest_x: last_dest_x, initial_dest_y: last_dest_y,
                                       :],
                                       (4, 2))

                self.inter_task_edges = self.inter_task_edges + [(tuple(source_nodes[x, y, :]), tuple(dst_node))
                                                                 for dst_node in dst_nodes]

    def add_inter_main_task_edges(self, all_graphs, graph_num):

        if self.use_connection_nodes:
            all_graphs = all_graphs[1:]

        source_nodes = np.reshape(np.array(all_graphs[graph_num - 1].nodes), (2 ** graph_num, 2 ** graph_num, 2))

        center_loc = (source_nodes.shape[0] - self.frame_size // 2) // 2
        source_nodes = source_nodes[center_loc: center_loc + self.frame_size // 2,
                       center_loc: center_loc + self.frame_size // 2, :]
        destination_nodes = np.reshape(np.array(all_graphs[graph_num].nodes), (self.frame_size,
                                                                               self.frame_size, 2))

        for x in range(source_nodes.shape[0]):
            for y in range(source_nodes.shape[1]):
                initial_dest_x = x * 2
                initial_dest_y = y * 2

                last_dest_x = initial_dest_x + 2
                last_dest_y = initial_dest_y + 2

                dst_nodes = np.reshape(destination_nodes[initial_dest_x: last_dest_x, initial_dest_y: last_dest_y,
                                       :],
                                       (4, 2))

                self.inter_task_edges = self.inter_task_edges + [(tuple(source_nodes[x, y, :]), tuple(dst_node))
                                                                 for dst_node in dst_nodes]

    def create_node_labels(self, coordinates, label_size, idx):

        lm_coords = list()
        valid_labels = list()
        # Note that the coordinates are saved in (h, w) convention. in order: LVID_top, LVID_bot, LVPW, IVS
        for i in range(4):

            try:
                if coordinates[i][0] != -1:

                    y = None

                    # Add the labels for the aux graphs
                    if not self.use_main_graph_only:
                        for graph_num in range(1, self.num_aux_graphs + 1):
                            bins = np.linspace(start=0, stop=self.frame_size, num=2 ** graph_num + 1)
                            transformed_coordinates = np.digitize(coordinates[i], bins=bins) - 1
                            transformed_coordinates = tuple(np.array(transformed_coordinates).T)

                            y_temp = np.zeros((2 ** graph_num, 2 ** graph_num))
                            y_temp[transformed_coordinates] = 1.0

                            y = torch.cat([y, torch.flatten(torch.tensor(y_temp, dtype=torch.float32))], dim=0) \
                                if graph_num != 1 else torch.flatten(torch.tensor(y_temp, dtype=torch.float32))

                    # Only add labels for the main graph if it's not using the 4-node coordinate implementation
                    transformed_coordinates = tuple(np.array(coordinates[i]).T.astype('int'))
                    y_temp = np.zeros((self.frame_size, self.frame_size))
                    y_temp[transformed_coordinates] = 1.0
                    y = torch.cat([y, torch.flatten(torch.tensor(y_temp, dtype=torch.float32))], dim=0) if y is not None else \
                        torch.flatten(torch.tensor(y_temp, dtype=torch.float32))

                    y = y.view((-1, 1))

                    lm_coords.append(y)
                    valid_labels.append(torch.ones_like(y))
                else:
                    lm_coords.append(torch.zeros(label_size))
                    valid_labels.append(torch.zeros(label_size))
            except IndexError:
                print("Index {} gave an error.".format(idx))
                lm_coords.append(torch.zeros(label_size))
                valid_labels.append(torch.zeros(label_size))

        output_labels = torch.cat(lm_coords, dim=1)
        valid_labels = torch.cat(valid_labels, dim=1)

        return output_labels, valid_labels


class DummyDataset(Dataset, ABC):
    def __init__(self,
                 data_dir,
                 data_info_file,
                 mode,
                 num_aux_graphs,
                 logger=None,
                 transform=None,
                 frame_size=128,
                 average_coords=None,
                 main_graph_type='grid',
                 aux_graph_type='grid',
                 use_coordinate_graph=False,
                 use_connection_nodes=False,
                 use_main_graph_only=False,
                 flip_p=0.0):

        super().__init__()

        if average_coords is None:
            # These numbers are obtained using the average_landmark_locations.py script
            self.average_coords = [[99.99, 112.57], [142.71, 90.67], [151.18, 86.25], [91.81, 117.91]]
        else:
            self.average_coords = average_coords

        # Create the graphs and the node types flag
        self.num_aux_graphs = num_aux_graphs
        self.frame_size = frame_size
        self.use_coordinate_graph = use_coordinate_graph
        self.use_connection_nodes = use_connection_nodes
        self.use_main_graph_only = use_main_graph_only

        self.graphs, self.node_type = self.create_graphs(main_graph_type, aux_graph_type)

        # Other required attributes
        self.mode = mode
        self.logger = logger
        self.transform = transform
        self.main_graph_type = main_graph_type
        self.flip_p = flip_p

    def __getitem__(self, idx):

        # ed_frame shape = (224,224),  transform to torch tensor with shape (1,1,resized_size,resized_size)
        orig_size = 224
        ed_frame = torch.randn((1, orig_size, orig_size))
        ed_frame = self.transform(ed_frame).unsqueeze(0)  # (1,1,frame_size,frame_size)

        # Extract landmark coordinates
        coords = self.extract_coords(None, orig_size)

        # Create PyG data using the prebuilt networkx graph
        g = from_networkx(self.graphs)

        # Add the echo frame to pyG data
        g.x = ed_frame

        # Create labels for each graph and add to PyG data
        g.y = torch.cat([self.create_node_labels(coord) for coord in coords], dim=1)
        g.valid_labels = torch.ones_like(g.y)

        # Add node type flag to PyG data
        g.node_type = torch.tensor(self.node_type)

        # Add initial location of nodes in the main graph and its labels
        if self.use_coordinate_graph and not self.use_main_graph_only:
            g.node_coords = torch.tensor(self.average_coords, dtype=torch.float32)
            g.node_coord_y = torch.tensor(coords, dtype=torch.float32)

        # Get the scale for each pixel in mm/pixel
        deltaX = 0.1
        deltaY = 0.1
        g.pix2mm_x = torch.tensor(deltaX * 10, dtype=torch.float32)  # in mm
        g.pix2mm_y = torch.tensor(deltaY * 10, dtype=torch.float32)  # in mm

        return g

    def __len__(self):

        return 100

    def extract_coords(self, df, orig_frame_size):

        # get all landmark coordinates, select the four we need
        LVIDd_coordinate = np.round(np.random.randint(low=0, high=self.frame_size,
                                                      size=4) * self.frame_size / orig_frame_size).astype(int)
        IVS_coordinates = np.round(np.random.randint(low=0, high=self.frame_size,
                                                     size=4) * self.frame_size / orig_frame_size).astype(int)
        LVPW_coordinates = np.round(np.random.randint(low=0, high=self.frame_size,
                                                      size=4) * self.frame_size / orig_frame_size).astype(int)

        # Note that the coordinates are saved in (h, w) convention
        coords = []
        coords.append([LVIDd_coordinate[1] - 1, LVIDd_coordinate[0] - 1])
        coords.append([LVIDd_coordinate[3] - 1, LVIDd_coordinate[2] - 1])
        coords.append([LVPW_coordinates[3] - 1, LVPW_coordinates[2] - 1])
        coords.append([IVS_coordinates[1] - 1, IVS_coordinates[0] - 1])
        coords = np.array(coords)

        return coords

    def create_graphs(self, main_graph_type, aux_graph_type):

        # List containing all graphs
        all_graphs = list()
        last_node = 0
        node_type = None
        self.inter_task_edges = list()

        if not self.use_main_graph_only:

            #  Graph connecting all aux graphs together
            if self.use_connection_nodes:
                connection_graph = nx.complete_graph(range(self.num_aux_graphs + 1))
                node_type = np.ones(connection_graph.number_of_nodes(), dtype=int)*2
                all_graphs.append(connection_graph)
                last_node = connection_graph.number_of_nodes()

            for graph_num in range(1, self.num_aux_graphs + 1):

                # Number of patches along each dim
                patches_along_dim = 2 ** graph_num

                # Create a grid graph
                if aux_graph_type == 'grid' or aux_graph_type == 'grid-diagonal':
                    aux_graph = nx.grid_graph(dim=[range(last_node, last_node + patches_along_dim),
                                                   range(last_node, last_node + patches_along_dim)])

                # Add the diagonal edges for grid-diagonal graphs
                if aux_graph_type == 'grid-diagonal':
                    # Code from https://stackoverflow.com/questions/55772715/how-to-create-8-cell-adjacency-map-for-a-diagonal-enabled-a-algorithm-with-the
                    edges = [((x, y), (x + 1, y + 1)) for x in range(last_node, last_node + patches_along_dim - 1) for y in
                             range(last_node, last_node + patches_along_dim - 1)] + \
                            [((x + 1, y), (x, y + 1)) for x in range(last_node, last_node + patches_along_dim - 1) for y in
                             range(last_node, last_node + patches_along_dim - 1)]
                    aux_graph.add_edges_from(edges)

                # Update flag indicating which nodes are connection (virtual) nodes
                node_type = np.hstack((node_type, np.zeros(aux_graph.number_of_nodes()))) if node_type is not None \
                    else np.zeros(aux_graph.number_of_nodes())

                # Add graphs together into a single graph
                all_graphs.append(aux_graph)
                last_node = list(all_graphs[-1].nodes)[-1][-1] + 1

            for graph_num in range(1, self.num_aux_graphs):
                self.add_inter_aux_task_edges(all_graphs, graph_num)

        # Create main grid graph
        if main_graph_type == 'grid' or main_graph_type == 'grid-diagonal':
            main_graph = nx.grid_graph(dim=[range(last_node, last_node + self.frame_size),
                                            range(last_node, last_node + self.frame_size)])

        # Add the diagonal edges for the grid-diagonal main graph
        if main_graph_type == 'grid-diagonal':
            # Code from https://stackoverflow.com/questions/55772715/how-to-create-8-cell-adjacency-map-for-a-diagonal-enabled-a-algorithm-with-the
            edges = [((x, y), (x + 1, y + 1)) for x in range(last_node, last_node + self.frame_size - 1) for y in
                     range(last_node, last_node + self.frame_size - 1)] + \
                    [((x + 1, y), (x, y + 1)) for x in range(last_node, last_node + self.frame_size - 1) for y in
                     range(last_node, last_node + self.frame_size - 1)]
            main_graph.add_edges_from(edges)

        # Add main graph to list of graphs
        all_graphs.append(main_graph)
        last_node = list(all_graphs[-1].nodes)[-1][-1] + 1
        node_type = np.hstack((node_type, np.zeros(main_graph.number_of_nodes()))) if node_type is not None \
            else np.zeros(main_graph.number_of_nodes())

        # Add edges between the finest aux graph and the main graph
        if not self.use_main_graph_only:
            self.add_inter_main_task_edges(all_graphs, self.num_aux_graphs)

            if self.use_connection_nodes:
                for graph_num in range(1, self.num_aux_graphs):
                    nodes = list(all_graphs[graph_num])
                    self.inter_task_edges = self.inter_task_edges + [(graph_num - 1, nodes[i]) for i in range(len(nodes))]

            # Create the coordinate graph (with only 4 nodes) for the main task
            if self.use_coordinate_graph:
                coord_graph = nx.complete_graph(range(last_node, last_node+4))

                node_type = np.hstack((node_type, np.ones(coord_graph.number_of_nodes())))

                all_graphs.append(coord_graph)

        # Consolidate all graphs
        for i, graph in enumerate(all_graphs):
            graphs = nx.compose(graphs, graph) if i != 0 else graph

        # Add the additional edges to graphs
        graphs.add_edges_from(self.inter_task_edges)

        return graphs, node_type

    def add_inter_aux_task_edges(self, all_graphs, graph_num):

        if self.use_connection_nodes:
            all_graphs = all_graphs[1:]

        source_nodes = np.reshape(np.array(all_graphs[graph_num - 1].nodes), (2 ** graph_num, 2 ** graph_num, 2))
        destination_nodes = np.reshape(np.array(all_graphs[graph_num].nodes), (2 ** (graph_num + 1),
                                                                               2 ** (graph_num + 1), 2))

        for x in range(source_nodes.shape[0]):
            for y in range(source_nodes.shape[1]):
                initial_dest_x = x * 2
                initial_dest_y = y * 2

                last_dest_x = initial_dest_x + 2
                last_dest_y = initial_dest_y + 2

                dst_nodes = np.reshape(destination_nodes[initial_dest_x: last_dest_x, initial_dest_y: last_dest_y,
                                       :],
                                       (4, 2))

                self.inter_task_edges = self.inter_task_edges + [(tuple(source_nodes[x, y, :]), tuple(dst_node))
                                                                 for dst_node in dst_nodes]

    def add_inter_main_task_edges(self, all_graphs, graph_num):

        if self.use_connection_nodes:
            all_graphs = all_graphs[1:]

        source_nodes = np.reshape(np.array(all_graphs[graph_num - 1].nodes), (2 ** graph_num, 2 ** graph_num, 2))

        center_loc = (source_nodes.shape[0] - self.frame_size//2) // 2
        source_nodes = source_nodes[center_loc: center_loc+self.frame_size//2,
                       center_loc: center_loc+self.frame_size//2, :]
        destination_nodes = np.reshape(np.array(all_graphs[graph_num].nodes), (self.frame_size,
                                                                               self.frame_size, 2))

        for x in range(source_nodes.shape[0]):
            for y in range(source_nodes.shape[1]):
                initial_dest_x = x * 2
                initial_dest_y = y * 2

                last_dest_x = initial_dest_x + 2
                last_dest_y = initial_dest_y + 2

                dst_nodes = np.reshape(destination_nodes[initial_dest_x: last_dest_x, initial_dest_y: last_dest_y,
                                       :],
                                       (4, 2))

                self.inter_task_edges = self.inter_task_edges + [(tuple(source_nodes[x, y, :]), tuple(dst_node))
                                                                 for dst_node in dst_nodes]

    def create_node_labels(self, coordinates):

        # Only a single coordinate is to be passed to this func

        y = None

        # Add the labels for the aux graphs
        if not self.use_main_graph_only:
            for graph_num in range(1, self.num_aux_graphs + 1):
                bins = np.linspace(start=0, stop=self.frame_size, num=2 ** graph_num + 1)
                transformed_coordinates = np.digitize(coordinates, bins=bins) - 1
                transformed_coordinates = tuple(np.array(transformed_coordinates).T)

                y_temp = np.zeros((2 ** graph_num, 2 ** graph_num))
                y_temp[transformed_coordinates] = 1.0

                y = torch.cat([y, torch.flatten(torch.tensor(y_temp, dtype=torch.float32))], dim=0) \
                    if graph_num != 1 else torch.flatten(torch.tensor(y_temp, dtype=torch.float32))

        # Only add labels for the main graph if it's not using the 4-node coordinate implementation
        transformed_coordinates = tuple(np.array(coordinates).T)
        y_temp = np.zeros((self.frame_size, self.frame_size))
        y_temp[transformed_coordinates] = 1.0
        y = torch.cat([y, torch.flatten(torch.tensor(y_temp, dtype=torch.float32))], dim=0) if y is not None else \
            torch.flatten(torch.tensor(y_temp, dtype=torch.float32))

        return y.view((-1, 1))


if __name__ == '__main__':
    from torchvision import transforms
    from torch_geometric.loader import DataListLoader, DataLoader

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
    ])
    dummy_dataset = LVLandmark(data_dir='../../data/Cleaned_LVPLAX2',
                              data_info_file='../../data/lv_plax2_cleaned_info_landmark_gt_filtered.csv',
                              mode='val', num_aux_graphs=1, transform=transform)

    g = dummy_dataset[0]
    print(g.x.shape, g.y.shape)

    use_data_parallel = True
    if use_data_parallel:
        dataloader = DataListLoader(dummy_dataset,
                                    batch_size=8,
                                    drop_last=True)

    else:
        dataloader = DataLoader(dummy_dataset,
                                batch_size=2,
                                drop_last=True)

    data_iter = iter(dataloader)
    g = next(data_iter)
    if use_data_parallel:
        x = torch.cat([graph.x for graph in g])
        y = torch.cat([graph.y for graph in g])
        SIUID = [graph.SIUID for graph in g]
        pix2mm_x = torch.hstack([graph.pix2mm_x for graph in g])
    print(x.shape, y.shape)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # for i in range(len(l)):
    #     plt.imshow(l[i])
    #     plt.savefig(f'{i}.png')
