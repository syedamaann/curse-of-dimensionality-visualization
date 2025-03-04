import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import pandas as pd
import math
import time

# Set random seed for reproducibility
np.random.seed(42)

# Generate data for multiple dimensions
def generate_data(n_points=200, max_dim=10):
    data = {}
    
    # 2D data: points on a circle
    theta = np.linspace(0, 2*np.pi, n_points)
    radius = 1 + 0.1 * np.random.randn(n_points)
    x_2d = radius * np.cos(theta)
    y_2d = radius * np.sin(theta)
    data['2d'] = {'x': x_2d, 'y': y_2d}
    
    # 3D data: points on a sphere
    phi = np.linspace(0, np.pi, n_points)
    theta = np.linspace(0, 2*np.pi, n_points)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    phi_flat = phi_grid.flatten()
    theta_flat = theta_grid.flatten()
    
    radius_3d = 1 + 0.1 * np.random.randn(len(phi_flat))
    x_3d = radius_3d * np.sin(phi_flat) * np.cos(theta_flat)
    y_3d = radius_3d * np.sin(phi_flat) * np.sin(theta_flat)
    z_3d = radius_3d * np.cos(phi_flat)
    data['3d'] = {'x': x_3d[:n_points], 'y': y_3d[:n_points], 'z': z_3d[:n_points]}
    
    # Generate data for dimensions 4 through max_dim
    for dim in range(4, max_dim + 1):
        # Create structured data in the current dimension
        dim_data = np.random.randn(n_points, dim)
        
        # Apply some structure to make dimensions related
        for i in range(1, dim):
            dim_data[:, i] = dim_data[:, 0] * i/dim + dim_data[:, i] * (1 - i/dim)
        
        # Project to 3D for visualization
        pca = PCA(n_components=3)
        projected = pca.fit_transform(dim_data)
        
        data[f'{dim}d'] = {
            'x': projected[:, 0], 
            'y': projected[:, 1], 
            'z': projected[:, 2],
            'raw': dim_data
        }
    
    return data

# Create visualization with dimension selector and explanations
def create_dimension_selector_visualization(max_dim=10):
    data = generate_data(max_dim=max_dim)
    
    # Create figure with initial 2D data
    fig = go.Figure()
    
    # Add a trace for each dimension
    for dim in range(2, max_dim + 1):
        dim_key = f'{dim}d'
        dim_data = data[dim_key]
        
        # Define color based on dimension
        r = max(0, int(255 * (1 - (dim-2)/8)))
        g = max(0, int(150 * (1 - (dim-2)/8)))
        b = min(255, int(100 + 155 * ((dim-2)/8)))
        color = f'rgb({r}, {g}, {b})'
        
        # For 2D, we need a special case
        if dim == 2:
            # 2D scatter plot
            scatter = go.Scatter(
                x=dim_data['x'],
                y=dim_data['y'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=color,
                    opacity=0.8
                ),
                name=f'{dim}D Representation',
                visible=True  # 2D is visible by default
            )
            fig.add_trace(scatter)
        else:
            # For dimensions 3 and higher
            # Add points
            scatter3d = go.Scatter3d(
                x=dim_data['x'],
                y=dim_data['y'],
                z=dim_data['z'],
                mode='markers',
                marker=dict(
                    size=5 + (dim-2),  # Points grow larger with dimensions
                    color=color,
                    opacity=0.8
                ),
                name=f'{dim}D Points',
                visible=False  # Hidden by default
            )
            fig.add_trace(scatter3d)
            
            # Add connections for dimensions 3 and higher
            if dim >= 3:
                # Calculate connections
                k = min(dim, 5)  # Number of connections per point
                
                if dim == 3:
                    # For 3D, use Euclidean distance in 3D space
                    points = np.column_stack((dim_data['x'], dim_data['y'], dim_data['z']))
                    distances = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2)
                else:
                    # For higher dimensions, use the raw high-dimensional data
                    distances = np.sum((dim_data['raw'][:, np.newaxis, :] - 
                                      dim_data['raw'][np.newaxis, :, :]) ** 2, axis=2)
                
                nearest_indices = np.argsort(distances, axis=1)[:, 1:k+1]
                
                lines_x, lines_y, lines_z = [], [], []
                for j in range(len(dim_data['x'])):
                    for idx in nearest_indices[j]:
                        lines_x.extend([dim_data['x'][j], dim_data['x'][idx], None])
                        lines_y.extend([dim_data['y'][j], dim_data['y'][idx], None])
                        lines_z.extend([dim_data['z'][j], dim_data['z'][idx], None])
                
                line_opacity = min(0.8, 0.3 + (dim-3)*0.1)
                line_color = f'rgba({r//2}, {g//2}, {b}, {line_opacity})'
                
                lines3d = go.Scatter3d(
                    x=lines_x,
                    y=lines_y,
                    z=lines_z,
                    mode='lines',
                    line=dict(
                        color=line_color,
                        width=1
                    ),
                    opacity=line_opacity,
                    name=f'{dim}D Connections',
                    visible=False  # Hidden by default
                )
                fig.add_trace(lines3d)
    
    # Create buttons for dimension selection
    buttons = []
    
    # 2D button (special case)
    button_2d = dict(
        method='update',
        args=[
            {'visible': [False] * len(fig.data)},  # Hide all traces
            {
                'title': '2D Representation',
                'annotations': [
                    {
                        'text': get_dimension_explanation(2),
                        'x': 0.5,
                        'y': 1.05,
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 14},
                        'align': 'center',
                        'bgcolor': 'rgba(255, 240, 240, 0.8)',
                        'bordercolor': 'rgba(255, 0, 0, 0.3)',
                        'borderwidth': 1,
                        'borderpad': 6
                    }
                ]
            }
        ],
        label='2D'
    )
    # Show only the first trace (2D)
    button_2d['args'][0]['visible'][0] = True
    buttons.append(button_2d)
    
    # Buttons for 3D and higher
    for dim in range(3, max_dim + 1):
        button = dict(
            method='update',
            args=[
                {'visible': [False] * len(fig.data)},  # Hide all traces
                {
                    'title': f'{dim}D Representation',
                    'annotations': [
                        {
                            'text': get_dimension_explanation(dim),
                            'x': 0.5,
                            'y': 1.05,
                            'xref': 'paper',
                            'yref': 'paper',
                            'showarrow': False,
                            'font': {'size': 14},
                            'align': 'center',
                            'bgcolor': f'rgba({255-dim*20}, {100+dim*10}, {100+dim*20}, 0.8)',
                            'bordercolor': f'rgba({150-dim*10}, {50+dim*5}, {100+dim*15}, 0.5)',
                            'borderwidth': 1,
                            'borderpad': 6
                        }
                    ]
                }
            ],
            label=f'{dim}D'
        )
        
        # Calculate which traces to show for this dimension
        if dim == 3:
            # For 3D, show the points and connections (indices 1 and 2)
            button['args'][0]['visible'][1] = True  # 3D points
            button['args'][0]['visible'][2] = True  # 3D connections
        else:
            # For higher dimensions, calculate the correct indices
            # Each dimension has 2 traces (points and connections)
            # 2D has 1 trace, 3D has 2 traces, so dim 4 starts at index 3
            point_index = 1 + (dim-3)*2
            connection_index = point_index + 1
            button['args'][0]['visible'][point_index] = True
            button['args'][0]['visible'][connection_index] = True
        
        buttons.append(button)
    
    # Add dimension selector menu
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                direction='right',
                active=0,
                x=0.5,
                y=1.15,
                xanchor='center',
                yanchor='top',
                buttons=buttons,
                bgcolor='#E2E2E2',
                bordercolor='#FFFFFF',
                font=dict(size=12)
            )
        ]
    )
    
    # Add title and labels
    fig.update_layout(
        title={
            'text': 'The Curse of Dimensionality: 2D Representation',  # Updated title
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        scene=dict(
            xaxis_title='X Dimension',
            yaxis_title='Y Dimension',
            zaxis_title='Z Dimension',
            aspectmode='cube'
        ),
        xaxis_title='X Dimension',
        yaxis_title='Y Dimension',
        margin=dict(t=100, b=100, l=100, r=100),  # Add margins for explanations
    )
    
    # Set camera position for 3D view
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    
    # Add initial explanation annotation
    fig.update_layout(
        annotations=[
            dict(
                text=get_dimension_explanation(2),
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
                align='center',
                bgcolor='rgba(255, 240, 240, 0.8)',
                bordercolor='rgba(255, 0, 0, 0.3)',
                borderwidth=1,
                borderpad=6
            )
        ]
    )
    
    return fig

# Function to get detailed explanations for each dimension
def get_dimension_explanation(dim):
    explanations = {
        2: """<b>The Curse of Dimensionality: 2D Representation</b><br>
            <b>The Many Dimensions of Reality:</b> Real-world phenomena often depend on many variables — hundreds, thousands, even millions.<br><br>
            <b>In 2D:</b> Points are arranged in a circle on a flat plane.<br>
            This is the simplest case where we can directly visualize all dimensions (x and y).<br>
            Our intuition works well here - distances between points are as we expect them to be.<br>
            Patterns are easy to detect (the circular arrangement is obvious).<br><br>
            <b>Human Intuition:</b> Our brains evolved to understand 2D and 3D spaces perfectly.""",
        
        3: """<b>The Curse of Dimensionality: 3D Representation</b><br>
            <b>The Many Dimensions of Reality:</b> As we add just one more dimension, visualization becomes more challenging.<br><br>
            <b>In 3D:</b> Points are arranged on the surface of a sphere.<br>
            We can still directly visualize all dimensions (x, y, and z), but need to rotate to see the full structure.<br>
            Our intuition still works reasonably well here.<br>
            The connections show relationships between nearby points on the sphere.<br><br>
            <b>Human Intuition:</b> This is the limit of what we can directly perceive in the physical world.""",
        
        4: """<b>The Curse of Dimensionality: 4D Representation</b><br>
            <b>The Many Dimensions of Reality:</b> At 4D, we've crossed a critical threshold where direct visualization is impossible.<br><br>
            <b>In 4D:</b> We can no longer directly visualize all dimensions - we must project to 3D.<br>
            The connections become crucial - they show which points are actually close in 4D space.<br>
            Points that appear far apart in our 3D view might actually be close in 4D.<br><br>
            <b>The Curse Begins:</b> Our intuition starts to break down.<br>
            - Distances between points become counterintuitive<br>
            - Patterns are harder to detect<br>
            - We must rely on mathematical projections rather than direct visualization""",
        
        5: """<b>The Curse of Dimensionality: 5D Representation</b><br>
            <b>The Many Dimensions of Reality:</b> At 5D, we're visualizing less than half of the actual dimensions.<br><br>
            <b>In 5D:</b> The projection to 3D loses significant information.<br>
            The connections are critical - they show relationships that exist in 5D but are distorted in our 3D view.<br>
            The structure becomes more complex as higher-dimensional relationships emerge.<br><br>
            <b>The Curse Intensifies:</b><br>
            - The "volume" of the space grows exponentially<br>
            - Points that seem clustered might be far apart in 5D<br>
            - Data becomes increasingly sparse relative to the space it occupies""",
        
        6: """<b>The Curse of Dimensionality: 6D Representation</b><br>
            <b>The Many Dimensions of Reality:</b> At 6D, we're only seeing half of the total dimensions.<br><br>
            <b>In 6D:</b> The connections reveal complex relationships that cannot be captured in the 3D projection alone.<br>
            Clusters that form in the visualization represent points that are close in 6D space.<br><br>
            <b>The Curse Deepens:</b><br>
            - The "corners" of the space dominate the volume<br>
            - Most points lie far from each other<br>
            - Nearest neighbor relationships become less meaningful<br>
            - Machine learning algorithms struggle with this sparsity""",
        
        7: """<b>The Curse of Dimensionality: 7D Representation</b><br>
            <b>The Many Dimensions of Reality:</b> At 7D, our intuition completely fails us.<br><br>
            <b>In 7D:</b> The network of connections becomes increasingly important.<br>
            Points with similar connection patterns have similar positions in 7D space.<br>
            The projection preserves only the most significant variance from the original 7D data.<br><br>
            <b>The Curse in Machine Learning:</b><br>
            - Need exponentially more data as dimensions increase<br>
            - Most statistical methods break down<br>
            - Optimization becomes harder (more local minima)<br>
            - "Distance" becomes less meaningful""",
        
        8: """<b>The Curse of Dimensionality: 8D Representation</b><br>
            <b>The Many Dimensions of Reality:</b> At 8D, we're in a realm where mathematics must replace intuition.<br><br>
            <b>In 8D:</b> The visualization shows only the three most significant dimensions out of eight.<br>
            The connection network reveals complex relationships that exist in 8D space.<br>
            Highly connected points represent regions of high density in the 8D space.<br><br>
            <b>Counter-Intuitive Properties:</b><br>
            - Almost all points lie at the "edge" of the space<br>
            - Random points tend to be nearly orthogonal (perpendicular)<br>
            - The "center" of the space is nearly empty<br>
            - Sampling becomes extremely inefficient""",
        
        9: """<b>The Curse of Dimensionality: 9D Representation</b><br>
            <b>The Many Dimensions of Reality:</b> At 9D, we're seeing a complex projection of high-dimensional data.<br><br>
            <b>In 9D:</b> The connections show which points are neighbors in 9D space despite their 3D projected positions.<br>
            The structure reveals patterns that would be impossible to visualize directly.<br><br>
            <b>Real-World Impact:</b><br>
            - In genomics, thousands of genes create a very high-dimensional space<br>
            - In computer vision, each pixel is a dimension (millions of dimensions)<br>
            - In economics, hundreds of variables interact in complex ways<br>
            - Our visualization is showing only a shadow of the true complexity""",
        
        10: """<b>The Curse of Dimensionality: 10D Representation</b><br>
            <b>The Many Dimensions of Reality:</b> At 10D, we're visualizing a highly complex space that defies human intuition.<br><br>
            <b>In 10D:</b> The connection network is crucial - it shows the true relationships in 10D space.<br>
            Points that appear close in the 3D projection might actually be far apart in 10D.<br><br>
            <b>The Full Curse of Dimensionality:</b><br>
            - Distances between points grow and become more uniform<br>
            - Patterns become nearly impossible to detect visually<br>
            - The volume of the space is astronomically larger than in 3D<br>
            - Most of the space is empty (data sparsity)<br>
            - Computational complexity explodes<br><br>
            <b>Breaking the Curse:</b> Dimensionality reduction, feature selection, and specialized algorithms<br>
            are essential for working with high-dimensional data."""
    }
    
    return explanations.get(dim, f"Explanation for {dim}D representation")

# Create animation that slowly rotates through each dimension
def create_animated_visualization(max_dim=10, frames_per_dim=100, rotation_speed=0.5):
    data = generate_data(max_dim=max_dim)
    frames = []
    
    # For each dimension
    for dim in range(2, max_dim + 1):
        dim_key = f'{dim}d'
        dim_data = data[dim_key]
        
        # Define color based on dimension
        r = max(0, int(255 * (1 - (dim-2)/8)))
        g = max(0, int(150 * (1 - (dim-2)/8)))
        b = min(255, int(100 + 155 * ((dim-2)/8)))
        color = f'rgb({r}, {g}, {b})'
        
        # For 2D, create a special case
        if dim == 2:
            for i in range(frames_per_dim):
                # For 2D, we'll just show the static 2D view
                frame = go.Frame(
                    data=[
                        go.Scatter(
                            x=dim_data['x'],
                            y=dim_data['y'],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=color,
                                opacity=0.8
                            ),
                            name=f'{dim}D Representation'
                        )
                    ],
                    layout=go.Layout(
                        title=f"The Curse of Dimensionality: {dim}D Representation",
                        annotations=[
                            dict(
                                text=get_dimension_explanation(dim),
                                x=0.5,
                                y=1.05,
                                xref="paper",
                                yref="paper",
                                showarrow=False,
                                font=dict(size=14),
                                align='center',
                                bgcolor='rgba(255, 240, 240, 0.8)',
                                bordercolor='rgba(255, 0, 0, 0.3)',
                                borderwidth=1,
                                borderpad=6
                            )
                        ]
                    ),
                    name=f"frame_{dim}_{i}"
                )
                frames.append(frame)
        else:
            # For 3D and higher dimensions
            # Calculate connections
            k = min(dim, 5)  # Number of connections per point
            
            if dim == 3:
                # For 3D, use Euclidean distance in 3D space
                points = np.column_stack((dim_data['x'], dim_data['y'], dim_data['z']))
                distances = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2)
            else:
                # For higher dimensions, use the raw high-dimensional data
                distances = np.sum((dim_data['raw'][:, np.newaxis, :] - 
                                  dim_data['raw'][np.newaxis, :, :]) ** 2, axis=2)
            
            nearest_indices = np.argsort(distances, axis=1)[:, 1:k+1]
            
            lines_x, lines_y, lines_z = [], [], []
            for j in range(len(dim_data['x'])):
                for idx in nearest_indices[j]:
                    lines_x.extend([dim_data['x'][j], dim_data['x'][idx], None])
                    lines_y.extend([dim_data['y'][j], dim_data['y'][idx], None])
                    lines_z.extend([dim_data['z'][j], dim_data['z'][idx], None])
            
            line_opacity = min(0.8, 0.3 + (dim-3)*0.1)
            line_color = f'rgba({r//2}, {g//2}, {b}, {line_opacity})'
            
            # Create frames with rotating camera for 3D and higher
            for i in range(frames_per_dim):
                # Calculate camera position for slow rotation
                angle = i * 2 * np.pi / frames_per_dim * rotation_speed
                eye_x = 1.5 * np.cos(angle)
                eye_y = 1.5 * np.sin(angle)
                eye_z = 1.5 * np.sin(angle/2) + 1.0
                
                frame = go.Frame(
                    data=[
                        go.Scatter3d(
                            x=dim_data['x'],
                            y=dim_data['y'],
                            z=dim_data['z'],
                            mode='markers',
                            marker=dict(
                                size=5 + (dim-2),  # Points grow larger with dimensions
                                color=color,
                                opacity=0.8
                            ),
                            name=f'{dim}D Points'
                        ),
                        go.Scatter3d(
                            x=lines_x,
                            y=lines_y,
                            z=lines_z,
                            mode='lines',
                            line=dict(
                                color=line_color,
                                width=1
                            ),
                            opacity=line_opacity,
                            name=f'{dim}D Connections'
                        )
                    ],
                    layout=go.Layout(
                        title=f"The Curse of Dimensionality: {dim}D Representation",
                        scene_camera=dict(
                            eye=dict(x=eye_x, y=eye_y, z=eye_z)
                        ),
                        annotations=[
                            dict(
                                text=get_dimension_explanation(dim),
                                x=0.5,
                                y=1.05,
                                xref="paper",
                                yref="paper",
                                showarrow=False,
                                font=dict(size=14),
                                align='center',
                                bgcolor=f'rgba({255-dim*20}, {100+dim*10}, {100+dim*20}, 0.8)',
                                bordercolor=f'rgba({150-dim*10}, {50+dim*5}, {100+dim*15}, 0.5)',
                                borderwidth=1,
                                borderpad=6
                            )
                        ]
                    ),
                    name=f"frame_{dim}_{i}"
                )
                frames.append(frame)
    
    # Create the initial figure with 2D data
    fig = go.Figure(
        data=[
            go.Scatter(
                x=data['2d']['x'],
                y=data['2d']['y'],
                mode='markers',
                marker=dict(
                    size=5,
                    color='rgb(255, 0, 0)',
                    opacity=0.8
                ),
                name='2D Representation'
            )
        ],
        frames=frames
    )
    
    # Add title and labels
    fig.update_layout(
        title={
            'text': 'The Curse of Dimensionality: 2D Representation',  # Updated title
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        scene=dict(
            xaxis_title='X Dimension',
            yaxis_title='Y Dimension',
            zaxis_title='Z Dimension',
            aspectmode='cube'
        ),
        xaxis_title='X Dimension',
        yaxis_title='Y Dimension',
        margin=dict(t=100, b=100, l=100, r=100),  # Add margins for explanations
    )
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=100, redraw=True),  # Slow animation (100ms per frame)
                            fromcurrent=True,
                            mode='immediate'
                        )]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ],
                direction='left',
                pad=dict(r=10, t=10),
                x=0.1,
                y=0,
                xanchor='right',
                yanchor='top'
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        method='animate',
                        args=[
                            [f"frame_{dim}_0"],  # Show first frame of each dimension
                            dict(
                                mode='immediate',
                                frame=dict(duration=100, redraw=True),
                                transition=dict(duration=0)
                            )
                        ],
                        label=f"{dim}D"
                    )
                    for dim in range(2, max_dim + 1)
                ],
                transition=dict(duration=0),
                x=0.1,
                y=0,
                currentvalue=dict(
                    font=dict(size=12),
                    prefix='Dimension: ',
                    visible=True,
                    xanchor='center'
                ),
                len=0.9
            )
        ]
    )
    
    # Add initial explanation annotation
    fig.update_layout(
        annotations=[
            dict(
                text=get_dimension_explanation(2),
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
                align='center',
                bgcolor='rgba(255, 240, 240, 0.8)',
                bordercolor='rgba(255, 0, 0, 0.3)',
                borderwidth=1,
                borderpad=6
            )
        ]
    )
    
    return fig

if __name__ == "__main__":
    # Create both visualizations
    print("Creating visualizations...")
    
    # Create the dimension selector visualization
    selector_fig = create_dimension_selector_visualization(max_dim=10)
    selector_fig.write_html("dimension_selector.html", auto_open=False)
    print("Dimension selector visualization saved as 'dimension_selector.html'")
    
    # Create the animated visualization with slow rotation
    animated_fig = create_animated_visualization(max_dim=10, frames_per_dim=60, rotation_speed=0.3)
    animated_fig.write_html("dimension_animation.html", auto_open=True)
    print("Animated visualization saved as 'dimension_animation.html'")
    
    print("Visualizations complete!")
