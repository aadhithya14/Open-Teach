/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using Oculus.Interaction.Input;
using UnityEditor;
using UnityEngine;

namespace Oculus.Interaction.Editor
{
    [CanEditMultipleObjects]
    [CustomEditor(typeof(HandWristOffset))]
    public class HandWristOffsetEditor : UnityEditor.Editor
    {
        private HandWristOffset _wristOffset;

        private SerializedProperty _offsetPositionProperty;
        private SerializedProperty _rotationProperty;
        private SerializedProperty _relativeTransformProperty;

        private Pose _cachedPose;

        private void Awake()
        {
            _wristOffset = target as HandWristOffset;

            _offsetPositionProperty = serializedObject.FindProperty("_offset");
            _rotationProperty = serializedObject.FindProperty("_rotation");
            _relativeTransformProperty = serializedObject.FindProperty("_relativeTransform");
        }

        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();

            _offsetPositionProperty.vector3Value = EditorGUILayout.Vector3Field("Offset", _offsetPositionProperty.vector3Value);
            Vector3 euler = EditorGUILayout.Vector3Field("Rotation", _rotationProperty.quaternionValue.eulerAngles);
            _rotationProperty.quaternionValue = Quaternion.Euler(euler);

            EditorGUILayout.PropertyField(_relativeTransformProperty);
            Transform gripPoint = _relativeTransformProperty.objectReferenceValue as Transform;
            if (gripPoint != null)
            {
                Pose offset;
                if (gripPoint != _wristOffset.transform)
                {
                    offset = _wristOffset.transform.RelativeOffset(gripPoint);
                }
                else
                {
                    offset = _wristOffset.transform.GetPose(Space.Self);
                }
                _rotationProperty.quaternionValue = FromOVRHandDataSource.WristFixupRotation * offset.rotation;
                _offsetPositionProperty.vector3Value = FromOVRHandDataSource.WristFixupRotation * offset.position;
            }

            serializedObject.ApplyModifiedProperties();
        }

        private void OnSceneGUI()
        {
            _cachedPose.position = _wristOffset.Offset;
            _cachedPose.rotation = _wristOffset.Rotation;

            Pose wristPose = _wristOffset.transform.GetPose();
            wristPose.rotation = wristPose.rotation * FromOVRHandDataSource.WristFixupRotation;
            _cachedPose.Postmultiply(wristPose);
            DrawAxis(_cachedPose);
        }

        private void DrawAxis(in Pose pose)
        {
            float scale = HandleUtility.GetHandleSize(pose.position);

#if UNITY_2020_2_OR_NEWER
            Handles.color = Color.red;
            Handles.DrawLine(pose.position, pose.position + pose.right * scale, EditorConstants.LINE_THICKNESS);
            Handles.color = Color.green;
            Handles.DrawLine(pose.position, pose.position + pose.up * scale, EditorConstants.LINE_THICKNESS);
            Handles.color = Color.blue;
            Handles.DrawLine(pose.position, pose.position + pose.forward * scale, EditorConstants.LINE_THICKNESS);
#else
            Handles.color = Color.red;
            Handles.DrawLine(pose.position, pose.position + pose.right * scale);
            Handles.color = Color.green;
            Handles.DrawLine(pose.position, pose.position + pose.up * scale);
            Handles.color = Color.blue;
            Handles.DrawLine(pose.position, pose.position + pose.forward * scale);
#endif
        }
    }
}
