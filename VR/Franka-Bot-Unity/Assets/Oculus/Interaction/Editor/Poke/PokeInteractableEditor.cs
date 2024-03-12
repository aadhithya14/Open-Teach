/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using UnityEditor;
using UnityEngine;

namespace Oculus.Interaction.Editor
{
    [CustomEditor(typeof(PokeInteractable))]
    public class PokeInteractableEditor : UnityEditor.Editor
    {
        private PokeInteractable _interactable;

        private SerializedProperty _proximityFieldProperty;
        private SerializedProperty _surfaceProperty;

        private static readonly float DRAW_RADIUS = 0.02f;

        private void Awake()
        {
            _interactable = target as PokeInteractable;

            _proximityFieldProperty = serializedObject.FindProperty("_proximityField");
            _surfaceProperty = serializedObject.FindProperty("_surface");
        }

        public void OnSceneGUI()
        {
            Handles.color = EditorConstants.PRIMARY_COLOR;
            Surfaces.PointablePlane plane = _surfaceProperty.objectReferenceValue as Surfaces.PointablePlane;

            if (plane == null)
            {
                // TODO support non-planar surfaces for this gizmo?
                return;
            }

            Transform triggerPlaneTransform = plane.transform;
            IProximityField proximityField = _proximityFieldProperty.objectReferenceValue as IProximityField;

            if (triggerPlaneTransform == null
                || proximityField == null)
            {
                return;
            }

            Vector3 touchPoint = triggerPlaneTransform.position - triggerPlaneTransform.forward * _interactable.MaxDistance;
            Vector3 proximalPoint = proximityField.ComputeClosestPoint(touchPoint);

            Handles.DrawSolidDisc(touchPoint, triggerPlaneTransform.forward, DRAW_RADIUS);

#if UNITY_2020_2_OR_NEWER
            Handles.DrawLine(touchPoint, proximalPoint, EditorConstants.LINE_THICKNESS);

            Handles.DrawLine(proximalPoint - triggerPlaneTransform.right * DRAW_RADIUS,
                proximalPoint + triggerPlaneTransform.right * DRAW_RADIUS, EditorConstants.LINE_THICKNESS);
            Handles.DrawLine(proximalPoint - triggerPlaneTransform.up * DRAW_RADIUS,
                proximalPoint + triggerPlaneTransform.up * DRAW_RADIUS, EditorConstants.LINE_THICKNESS);
#else
            Handles.DrawLine(touchPoint, proximalPoint);

            Handles.DrawLine(proximalPoint - triggerPlaneTransform.right * DRAW_RADIUS,
                proximalPoint + triggerPlaneTransform.right * DRAW_RADIUS);
            Handles.DrawLine(proximalPoint - triggerPlaneTransform.up * DRAW_RADIUS,
                proximalPoint + triggerPlaneTransform.up * DRAW_RADIUS);
#endif

        }
    }
}
