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

namespace Oculus.Interaction.HandGrab.Editor
{
    [CanEditMultipleObjects]
    [CustomEditor(typeof(HandGrabInteractable))]
    public class HandGrabInteractableEditor : UnityEditor.Editor
    {
        private HandGrabInteractable _interactable;

        private void Awake()
        {
            _interactable = target as HandGrabInteractable;
        }

        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();

            DrawGrabPosesMenu();
            GUILayout.Space(20f);
            DrawGenerationMenu();
        }

        private void DrawGrabPosesMenu()
        {
            if (GUILayout.Button("Refresh HandGrab Pose"))
            {
                _interactable.HandGrabPoses.Clear();
                HandGrabPose[] handGrabPoses = _interactable.GetComponentsInChildren<HandGrabPose>();
                _interactable.HandGrabPoses.AddRange(handGrabPoses);
            }

            if (GUILayout.Button("Add HandGrab Pose"))
            {
                if (_interactable.HandGrabPoses.Count > 0)
                {
                    AddHandGrabPose(_interactable.HandGrabPoses[0]);
                }
                else
                {
                    AddHandGrabPose();
                }
            }

            if (GUILayout.Button("Replicate Default Scaled HandGrab Pose"))
            {
                if (_interactable.HandGrabPoses.Count > 0)
                {
                    AddHandGrabPose(_interactable.HandGrabPoses[0], 0.8f);
                    AddHandGrabPose(_interactable.HandGrabPoses[0], 1.2f);
                }
                else
                {
                    Debug.LogError("You have to provide a default HandGrabPose first!");
                }
            }
        }

        private void AddHandGrabPose(HandGrabPose copy = null, float? scale = null)
        {
            HandGrabPose point = _interactable.CreatePoint();
            if (copy != null)
            {
                HandGrabPoseEditor.CloneHandGrabPose(copy, point);
                if (scale.HasValue)
                {
                    HandGrabPoseData scaledData = point.SaveData();
                    scaledData.scale = scale.Value;
                    point.LoadData(scaledData, copy.RelativeTo);
                }
            }
            _interactable.HandGrabPoses.Add(point);
        }

        private void DrawGenerationMenu()
        {
            if (GUILayout.Button("Create Mirrored HandGrabInteractable"))
            {
                HandGrabInteractable mirrorInteractable =
                    HandGrabInteractable.Create(_interactable.RelativeTo,
                        $"{_interactable.gameObject.name}_mirror");

                HandGrabInteractableData data = _interactable.SaveData();
                data.poses = null;
                mirrorInteractable.LoadData(data);

                foreach (HandGrabPose point in _interactable.HandGrabPoses)
                {
                    HandGrabPose mirrorPoint = mirrorInteractable.CreatePoint();
                    HandGrabPoseEditor.Mirror(point, mirrorPoint);
                    mirrorPoint.transform.SetParent(mirrorInteractable.transform);
                    mirrorInteractable.HandGrabPoses.Add(mirrorPoint);
                }
            }
        }
    }
}
