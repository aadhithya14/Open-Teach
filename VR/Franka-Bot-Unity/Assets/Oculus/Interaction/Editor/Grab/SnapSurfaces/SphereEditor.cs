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
    [CustomEditor(typeof(SphereSurface))]
    [CanEditMultipleObjects]
    public class SphereEditor : UnityEditor.Editor
    {
        private SphereBoundsHandle _sphereHandle = new SphereBoundsHandle();
        private SphereSurface _surface;

        private void OnEnable()
        {
            _sphereHandle.SetColor(EditorConstants.PRIMARY_COLOR);
            _sphereHandle.midpointHandleDrawFunction = null;

            _surface = (target as SphereSurface);
        }

        public void OnSceneGUI()
        {
            DrawCentre(_surface);
            Handles.color = Color.white;
            DrawSphereEditor(_surface);

            if (Event.current.type == EventType.Repaint)
            {
                DrawSurfaceVolume(_surface);
            }
        }

        private void DrawCentre(SphereSurface surface)
        {
            EditorGUI.BeginChangeCheck();
            Quaternion handleRotation = (surface.RelativeTo ?? surface.transform).rotation;
            Vector3 centrePosition = Handles.PositionHandle(surface.Centre, handleRotation);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(surface, "Change Centre Sphere Position");
                surface.Centre = centrePosition;
            }
        }

        private void DrawSurfaceVolume(SphereSurface surface)
        {
            Handles.color = EditorConstants.PRIMARY_COLOR;
            Vector3 startLine = surface.Centre;
            Vector3 endLine = startLine + surface.Rotation * Vector3.forward * surface.Radius;
            Handles.DrawDottedLine(startLine, endLine, 5);
        }
        private void DrawSphereEditor(SphereSurface surface)
        {
            _sphereHandle.radius = surface.Radius;
            _sphereHandle.center = surface.Centre;

            EditorGUI.BeginChangeCheck();
            _sphereHandle.DrawHandle();
        }
    }
}
