/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using Oculus.Interaction.Editor;
using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

namespace Oculus.Interaction.HandGrab.SnapSurfaces.Editor
{
    [CustomEditor(typeof(CylinderSurface))]
    [CanEditMultipleObjects]
    public class CylinderEditor : UnityEditor.Editor
    {
        private const float DRAW_SURFACE_ANGULAR_RESOLUTION = 5f;

        private ArcHandle _arcHandle = new ArcHandle();
        private Vector3[] _surfaceEdges;

        CylinderSurface _surface;

        private void OnEnable()
        {
            _arcHandle.SetColorWithRadiusHandle(EditorConstants.PRIMARY_COLOR, 0f);
            _surface = (target as CylinderSurface);
        }

        public void OnSceneGUI()
        {
            DrawEndsCaps(_surface);
            DrawArcEditor(_surface);
            if (Event.current.type == EventType.Repaint)
            {
                DrawSurfaceVolume(_surface);
            }
        }

        private void DrawEndsCaps(CylinderSurface surface)
        {
            EditorGUI.BeginChangeCheck();

            Quaternion handleRotation = (surface.RelativeTo ?? surface.transform).rotation;

            Vector3 startPosition = Handles.PositionHandle(surface.StartPoint, handleRotation);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(surface, "Change Start Cylinder Position");
                surface.StartPoint = startPosition;
            }
            EditorGUI.BeginChangeCheck();
            Vector3 endPosition = Handles.PositionHandle(surface.EndPoint, handleRotation);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(surface, "Change Start Cylinder Position");
                surface.EndPoint = endPosition;
            }
        }

        private void DrawSurfaceVolume(CylinderSurface surface)
        {
            Vector3 start = surface.StartPoint;
            Vector3 end = surface.EndPoint;
            float radius = surface.Radius;

            Handles.color = EditorConstants.PRIMARY_COLOR;
            Handles.DrawWireArc(end,
                surface.Direction,
                surface.StartAngleDir,
                surface.Angle,
                radius);

            Handles.DrawLine(start, end);
            Handles.DrawLine(start, start + surface.StartAngleDir * radius);
            Handles.DrawLine(start, start + surface.EndAngleDir * radius);
            Handles.DrawLine(end, end + surface.StartAngleDir * radius);
            Handles.DrawLine(end, end + surface.EndAngleDir * radius);

            int edgePoints = Mathf.CeilToInt((2 * surface.Angle) / DRAW_SURFACE_ANGULAR_RESOLUTION) + 3;
            if (_surfaceEdges == null
                || _surfaceEdges.Length != edgePoints)
            {
                _surfaceEdges = new Vector3[edgePoints];
            }

            Handles.color = EditorConstants.PRIMARY_COLOR_DISABLED;
            int i = 0;
            for (float angle = 0f; angle < surface.Angle; angle += DRAW_SURFACE_ANGULAR_RESOLUTION)
            {
                Vector3 direction = Quaternion.AngleAxis(angle, surface.Direction) * surface.StartAngleDir;
                _surfaceEdges[i++] = start + direction * radius;
                _surfaceEdges[i++] = end + direction * radius;
            }
            _surfaceEdges[i++] = start + surface.EndAngleDir * radius;
            _surfaceEdges[i++] = end + surface.EndAngleDir * radius;
            Handles.DrawPolyLine(_surfaceEdges);
        }

        private void DrawArcEditor(CylinderSurface surface)
        {
            float radius = surface.Radius;
            _arcHandle.angle = surface.Angle;
            _arcHandle.radius = radius;

            Matrix4x4 handleMatrix = Matrix4x4.TRS(
                surface.StartPoint,
                Quaternion.LookRotation(surface.StartAngleDir, surface.Direction),
                Vector3.one
            );
            using (new Handles.DrawingScope(handleMatrix))
            {
                EditorGUI.BeginChangeCheck();
                _arcHandle.DrawHandle();
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RecordObject(surface, "Change Cylinder Properties");
                    surface.Angle = _arcHandle.angle;
                    radius = _arcHandle.radius;
                }
            }
        }
    }
}
