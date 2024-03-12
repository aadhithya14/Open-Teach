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
using Oculus.Interaction.HandGrab.Visuals;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using Oculus.Interaction.Editor;

namespace Oculus.Interaction.HandGrab.Editor
{
    [CustomEditor(typeof(HandGrabPose))]
    public class HandGrabPoseEditor : UnityEditor.Editor
    {
        private HandGrabPose _handGrabPose;

        private HandGhostProvider _ghostVisualsProvider;
        private HandGhost _handGhost;
        private HandPuppet _ghostPuppet;
        private Handedness _lastHandedness;

        private bool _editingFingers;

        private SerializedProperty _handPoseProperty;

        private const float GIZMO_SCALE = 0.005f;

        private void Awake()
        {
            _handGrabPose = target as HandGrabPose;
            _handPoseProperty = serializedObject.FindProperty("_handPose");
            AssignMissingGhostProvider();
        }

        private void OnDestroy()
        {
            DestroyGhost();
        }

        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();

            if (_handGrabPose.HandPose != null
                && _handPoseProperty != null)
            {
                EditorGUILayout.PropertyField(_handPoseProperty);
                EditorGUILayout.Space();
                DrawGhostMenu(_handGrabPose.HandPose, false);
            }
            else if (_handGhost != null)
            {
                DestroyGhost();
            }
            serializedObject.ApplyModifiedProperties();
        }

        private void DrawGhostMenu(HandPose handPose, bool forceCreate)
        {
            GUIStyle boldStyle = new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold };
            EditorGUILayout.LabelField("Interactive Edition", boldStyle);

            HandGhostProvider provider = EditorGUILayout.ObjectField("Ghost Provider", _ghostVisualsProvider, typeof(HandGhostProvider), false) as HandGhostProvider;
            if (forceCreate
                || provider != _ghostVisualsProvider
                || _handGhost == null
                || _lastHandedness != handPose.Handedness)
            {
                RegenerateGhost(provider);
            }
            _lastHandedness = handPose.Handedness;

            if (_handGrabPose.SnapSurface == null)
            {
                _editingFingers = true;
            }
            else if (GUILayout.Button(_editingFingers ? "Follow Surface" : "Edit fingers"))
            {
                _editingFingers = !_editingFingers;
            }
        }

        public void OnSceneGUI()
        {
            if (SceneView.currentDrawingSceneView == null
                || _handGhost == null)
            {
                return;
            }

            if (_editingFingers)
            {
                GhostEditFingers();
                if (AnyPuppetBoneChanged())
                {
                    TransferGhostBoneRotations();
                }
            }
            else
            {
                GhostFollowSurface();
            }
        }

        #region generation
        /// <summary>
        /// Generates a new HandGrabPoseData that mirrors the provided one. Left hand becomes right hand and vice-versa.
        /// The mirror axis is defined by the surface of the snap point, if any, if none a best-guess is provided
        /// but note that it can then moved manually in the editor.
        /// </summary>
        /// <param name="originalPoint">The point to mirror</param>
        /// <param name="originalPoint">The target HandGrabPose to set as mirrored of the originalPoint</param>
        public static void Mirror(HandGrabPose originalPoint, HandGrabPose mirrorPoint)
        {
            HandPose handPose = originalPoint.HandPose;

            Handedness oppositeHandedness = handPose.Handedness == Handedness.Left ? Handedness.Right : Handedness.Left;

            HandGrabPoseData mirrorData = originalPoint.SaveData();
            mirrorData.handPose.Handedness = oppositeHandedness;

            if (originalPoint.SnapSurface != null)
            {
                mirrorData.gripPose = originalPoint.SnapSurface.MirrorPose(mirrorData.gripPose);
            }
            else
            {
                mirrorData.gripPose = mirrorData.gripPose.MirrorPoseRotation(Vector3.forward, Vector3.up);
                Vector3 translation = Vector3.Project(mirrorData.gripPose.position, Vector3.right);
                mirrorData.gripPose.position = mirrorData.gripPose.position - 2f * translation;
            }

            mirrorPoint.LoadData(mirrorData, originalPoint.RelativeTo);
            if (originalPoint.SnapSurface != null)
            {
                SnapSurfaces.ISnapSurface mirroredSurface = originalPoint.SnapSurface.CreateMirroredSurface(mirrorPoint.gameObject);
                mirrorPoint.InjectOptionalSurface(mirroredSurface);
            }
        }

        public static void CloneHandGrabPose(HandGrabPose originalPoint, HandGrabPose targetPoint)
        {
            HandGrabPoseData mirrorData = originalPoint.SaveData();
            targetPoint.LoadData(mirrorData, originalPoint.RelativeTo);
            if (originalPoint.SnapSurface != null)
            {
                SnapSurfaces.ISnapSurface mirroredSurface = originalPoint.SnapSurface.CreateDuplicatedSurface(targetPoint.gameObject);
                targetPoint.InjectOptionalSurface(mirroredSurface);
            }
        }
        #endregion

        #region ghost

        private void AssignMissingGhostProvider()
        {
            if (_ghostVisualsProvider != null)
            {
                return;
            }

            HandGhostProviderUtils.TryGetDefaultProvider(out _ghostVisualsProvider);
        }

        private void RegenerateGhost(HandGhostProvider provider)
        {
            _ghostVisualsProvider = provider;
            DestroyGhost();
            CreateGhost();
        }

        private void CreateGhost()
        {
            if (_ghostVisualsProvider == null)
            {
                return;
            }

            HandGhost ghostPrototype = _ghostVisualsProvider.GetHand(_handGrabPose.HandPose.Handedness);
            _handGhost = GameObject.Instantiate(ghostPrototype, _handGrabPose.transform);
            _handGhost.gameObject.hideFlags = HideFlags.HideAndDontSave;
            _handGhost.SetPose(_handGrabPose);
            _ghostPuppet = _handGhost.GetComponent<HandPuppet>();
        }

        private void DestroyGhost()
        {
            if (_handGhost == null)
            {
                return;
            }
            GameObject.DestroyImmediate(_handGhost.gameObject);
            _handGhost = null;
        }

        private void GhostFollowSurface()
        {
            if (_handGhost == null)
            {
                return;
            }

            Pose ghostTargetPose = _handGrabPose.RelativeGrip;

            if (_handGrabPose.SnapSurface != null)
            {
                Vector3 mousePosition = Event.current.mousePosition;
                Ray ray = HandleUtility.GUIPointToWorldRay(mousePosition);
                Pose recorderPose = _handGrabPose.transform.GetPose();
                if (_handGrabPose.SnapSurface.CalculateBestPoseAtSurface(ray, recorderPose, out Pose target))
                {
                    ghostTargetPose.position = _handGrabPose.RelativeTo.InverseTransformPoint(target.position);
                    ghostTargetPose.rotation = Quaternion.Inverse(_handGrabPose.RelativeTo.rotation) * target.rotation;
                }
            }

            _handGhost.SetRootPose(ghostTargetPose, _handGrabPose.RelativeTo);
        }

        private void GhostEditFingers()
        {
            HandPuppet puppet = _handGhost.GetComponent<HandPuppet>();
            if (puppet != null && puppet.JointMaps != null)
            {
                DrawBonesRotator(puppet.JointMaps);
            }
        }

        private void TransferGhostBoneRotations()
        {
            HandPose poseTransfer = _handGrabPose.HandPose;
            _ghostPuppet.CopyCachedJoints(ref poseTransfer);
            EditorUtility.SetDirty(_handGrabPose);
        }

        private void DrawBonesRotator(List<HandJointMap> bones)
        {
            foreach (HandJointMap bone in bones)
            {
                HandFinger finger = FingersMetadata.JOINT_TO_FINGER[(int)bone.id];

                if (_handGrabPose.HandPose.FingersFreedom[(int)finger] == JointFreedom.Free)
                {
                    continue;
                }

                Handles.color = EditorConstants.PRIMARY_COLOR;
                Quaternion rotation = Handles.Disc(bone.transform.rotation, bone.transform.position,
                    bone.transform.forward, GIZMO_SCALE, false, 0);

                int metadataIndex = FingersMetadata.HandJointIdToIndex(bone.id);
                if (FingersMetadata.HAND_JOINT_CAN_SPREAD[metadataIndex])
                {
                    Handles.color = EditorConstants.SECONDARY_COLOR;
                    rotation = Handles.Disc(rotation, bone.transform.position,
                        bone.transform.up, GIZMO_SCALE, false, 0);
                }

                if (bone.transform.rotation != rotation)
                {
                    Undo.RecordObject(bone.transform, "Bone Rotation");
                    bone.transform.rotation = rotation;
                }

            }
        }

        /// <summary>
        /// Detects if we have moved the transforms of the fingers so the data can be kept up to date.
        /// To be used in Edit-mode.
        /// </summary>
        /// <returns>True if any of the fingers has moved from the previous frame.</returns>
        private bool AnyPuppetBoneChanged()
        {
            bool hasChanged = false;
            foreach (HandJointMap bone in _ghostPuppet.JointMaps)
            {
                if (bone.transform.hasChanged)
                {
                    bone.transform.hasChanged = false;
                    hasChanged = true;
                }
            }
            return hasChanged;
        }

        #endregion

    }
}
