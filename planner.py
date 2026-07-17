import torch
import torch.nn as nn
import torchvision.models as models


class OpenPilotStylePlanner(nn.Module):
    def __init__(self, future_steps=20, visual_features_dim=512, hidden_dim=256):
        super(OpenPilotStylePlanner, self).__init__()

        self.future_steps = (
            future_steps  # e.g., 20 frames into the future (approx. 2 seconds)
        )
        self.hidden_dim = hidden_dim

        # 1. Spatial Backbone: Extracts visual features from raw camera frames
        # In production, this can be swapped with a Vision Transformer (ViT) or EfficientNet
        base_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.visual_encoder = nn.Sequential(
            *list(base_resnet.children())[:-1]
        )  # Strip final FC layer

        # Linear layer to project visual features to standard embedding size
        self.visual_projector = nn.Linear(
            base_resnet.fc.in_features, visual_features_dim
        )

        # 2. Kinematics Encoder: Encodes vehicle states (e.g., current speed, steering angle)
        # Input shape: [batch, 2] -> (speed, steering_angle)
        self.kinematics_encoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )

        # 3. Temporal Engine: Blends spatial vision, vehicle memory, and kinematics over time
        # Combined feature size = visual_features_dim + kinematics_dim
        combined_input_dim = visual_features_dim + 64
        self.temporal_gru = nn.GRU(
            input_size=combined_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # 4. Trajectory Trait Head: Outputs (X, Y) coordinate points for the future horizon
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, future_steps * 2),  # Multiplied by 2 for X and Y coordinates
        )

    def forward(self, video_frames, vehicle_states):
        """
        Args:
            video_frames: Tensor of shape [Batch, Sequence_Length, Channels, Height, Width]
            vehicle_states: Tensor of shape [Batch, Sequence_Length, 2] (Velocity & Steering)
        Returns:
            predicted_trajectory: Tensor of shape [Batch, Future_Steps, 2] (X, Y path points)
        """
        batch_size, seq_len, c, h, w = video_frames.size()

        # Collapse batch and sequence dimensions to process images through ConvNet simultaneously
        flat_frames = video_frames.view(batch_size * seq_len, c, h, w)
        visual_features = self.visual_encoder(flat_frames)
        visual_features = torch.flatten(visual_features, 1)
        visual_features = self.visual_projector(visual_features)

        # Reshape back to sequence form: [Batch, Sequence_Length, Visual_Features_Dim]
        visual_features = visual_features.view(batch_size, seq_len, -1)

        # Encode continuous vehicle kinematics
        kinematic_features = self.kinematics_encoder(vehicle_states)

        # Concatenate visual cues and vehicle physics along the feature dimension
        # Resulting shape: [Batch, Sequence_Length, Combined_Input_Dim]
        combined_features = torch.cat((visual_features, kinematic_features), dim=2)

        # Pass the temporal sequence through the GRU
        # gru_out shape: [Batch, Sequence_Length, Hidden_Dim]
        gru_out, _ = self.temporal_gru(combined_features)

        # Use the hidden state of the *most recent* frame to predict the future trajectory
        latest_memory_token = gru_out[:, -1, :]

        # Generate raw 1D trajectory coordinates
        raw_trajectory = self.trajectory_head(latest_memory_token)

        # Reshape to 2D trajectory coordinates: [Batch, Future_Steps, 2 (X, Y)]
        predicted_trajectory = raw_trajectory.view(batch_size, self.future_steps, 2)

        return predicted_trajectory


# --- Quick Verification Test Loop ---
if __name__ == "__main__":
    # # Model Hyperparameters
    # B, T, C, H, W = 4, 10, 3, 224, 224  # Batch size 4, 10-frame past context history

    # # Initialize the End-to-End network
    # model = OpenPilotStylePlanner(future_steps=20)
    # model.eval()  # Set to evaluation mode

    # # Generate mock tensors simulating camera inputs and car CAN bus speeds
    # dummy_video = torch.randn(B, T, C, H, W)
    # dummy_kinematics = torch.randn(B, T, 2)  # Simulating (velocity, steering angle)

    # # Compute output path
    # with torch.no_grad():
    #     trajectory = model(dummy_video, dummy_kinematics)

    # print(f"Input Video Batch Shape: {dummy_video.shape}")
    # print(
    #     f"Generated Future Trajectory Shape: {trajectory.shape} -> [Batch, Steps, (X,Y)]"
    # )

    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    # from collections import Counter, defaultdict

    # import onnx

    # model = onnx.load("data/driving_supercombo.onnx")
    # graph = model.graph

    # print("Inputs:")
    # for x in graph.input:
    #     print(" ", x.name)

    # print("\nOutputs:")
    # for x in graph.output:
    #     print(" ", x.name)

    # print("\nInitializers / weights:", len(graph.initializer))
    # for w in graph.initializer[:20]:
    #     print(" ", w.name, list(w.dims), w.data_type)

    # print("\nOperator histogram:")
    # ops = Counter(n.op_type for n in graph.node)
    # for op, c in ops.most_common():
    #     print(f"{op:24s} {c}")

    # print("\nPossible transformer-related nodes:")
    # for i, n in enumerate(graph.node):
    #     if n.op_type in {
    #         "MatMul",
    #         "Gemm",
    #         "Softmax",
    #         "LayerNormalization",
    #         "Reshape",
    #         "Transpose",
    #         "ReduceMean",
    #         "Div",
    #         "Sqrt",
    #     }:
    #         print(f"{i:5d} {n.op_type:20s} {n.name}")
    #         print("      in :", list(n.input))
    #         print("      out:", list(n.output))

    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    from collections import defaultdict

    import onnx

    model = onnx.load("data/driving_supercombo.onnx")

    groups = defaultdict(list)

    for init in model.graph.initializer:
        name = init.name
        parts = name.split(".")

        # Group by useful module depth
        if "stages" in parts and "blocks" in parts:
            si = parts.index("stages")
            bi = parts.index("blocks")
            key = ".".join(parts[: bi + 2])  # up to stages.X.blocks.Y
        elif "transformer" in parts:
            ti = parts.index("transformer")
            key = ".".join(parts[: ti + 2])  # up to transformer.N
        elif "hydra" in parts:
            hi = parts.index("hydra")
            key = ".".join(parts[: hi + 1])
        elif "summarizer" in parts:
            si = parts.index("summarizer")
            key = ".".join(parts[: si + 1])
        else:
            key = ".".join(parts[:4])

        groups[key].append((name, list(init.dims)))

    for key in sorted(groups):
        print("\n" + key)
        for name, shape in groups[key]:
            print("  ", name, shape)
