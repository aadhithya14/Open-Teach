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
    [CustomEditor(typeof(CircleProximityField))]
    public class CircleProximityFieldEditor : UnityEditor.Editor
    {
        private SerializedProperty _transformProperty;
        private SerializedProperty _radiusProperty;

        private void Awake()
        {
            _transformProperty = serializedObject.FindProperty("_transform");
            _radiusProperty = serializedObject.FindProperty("_radius");
        }

        public void OnSceneGUI()
        {
            Handles.color = EditorConstants.PRIMARY_COLOR;

            Transform transform = _transformProperty.objectReferenceValue as Transform;
            float radius = _radiusProperty.floatValue * transform.lossyScale.x;
#if UNITY_2020_2_OR_NEWER
            Handles.DrawWireDisc(transform.position, -transform.forward, radius, EditorConstants.LINE_THICKNESS);
#else
            Handles.DrawWireDisc(transform.position, -transform.forward, radius);
#endif
        }
    }
}
