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
    [CustomEditor(typeof(BoxProximityField))]
    public class BoxProximityFieldEditor : UnityEditor.Editor
    {
        private SerializedProperty _boxTransformProperty;

        private void Awake()
        {
            _boxTransformProperty = serializedObject.FindProperty("_boxTransform");
        }

        public void OnSceneGUI()
        {
            Handles.color = EditorConstants.PRIMARY_COLOR;

            Transform boxTransform = _boxTransformProperty.objectReferenceValue as Transform;

            if (boxTransform != null)
            {
                using (new Handles.DrawingScope(boxTransform.localToWorldMatrix))
                {
                    Handles.DrawWireCube(Vector3.zero, Vector3.one);
                }
            }
        }
    }
}
