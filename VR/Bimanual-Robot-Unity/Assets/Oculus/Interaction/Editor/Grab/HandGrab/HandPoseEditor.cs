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
using Oculus.Interaction.Input;
using System;
using UnityEditor;
using UnityEngine;

namespace Oculus.Interaction.HandGrab.Editor
{
    [CustomPropertyDrawer(typeof(HandPose))]
    public class HandPoseEditor : PropertyDrawer
    {
        private bool _foldedFreedom = true;

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            if (_foldedFreedom)
            {
                return EditorConstants.ROW_HEIGHT * (Constants.NUM_FINGERS + 3);
            }
            else
            {
                return EditorConstants.ROW_HEIGHT * 3;
            }
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            EditorGUI.BeginProperty(position, label, property);
            Rect labelPos = EditorGUI.PrefixLabel(position, GUIUtility.GetControlID(FocusType.Passive), label);
            EditorGUI.indentLevel++;

            Rect rowRect = new Rect(position.x, labelPos.y + EditorConstants.ROW_HEIGHT, position.width, EditorConstants.ROW_HEIGHT);
            DrawFlagProperty<Handedness>(property, rowRect, "Handedness:", "_handedness", false);
            rowRect.y += EditorConstants.ROW_HEIGHT;
            DrawFingersFreedomMenu(property, rowRect);

            EditorGUI.indentLevel--;
            EditorGUI.EndProperty();
        }

        private Rect DrawFingersFreedomMenu(SerializedProperty property, Rect position)
        {
            _foldedFreedom = EditorGUI.Foldout(position, _foldedFreedom, "Fingers Freedom", true);
            position.y += EditorConstants.ROW_HEIGHT;
            if (_foldedFreedom)
            {
                SerializedProperty fingersFreedom = property.FindPropertyRelative("_fingersFreedom");
                EditorGUI.indentLevel++;
                for (int i = 0; i < Constants.NUM_FINGERS; i++)
                {
                    SerializedProperty finger = fingersFreedom.GetArrayElementAtIndex(i);
                    HandFinger fingerID = (HandFinger)i;
                    JointFreedom current = (JointFreedom)finger.intValue;
                    JointFreedom selected = (JointFreedom)EditorGUI.EnumPopup(position, $"{fingerID}: ", current);
                    finger.intValue = (int)selected;
                    position.y += EditorConstants.ROW_HEIGHT;
                }
                EditorGUI.indentLevel--;
            }

            return position;
        }

        private void DrawFlagProperty<TEnum>(SerializedProperty parentProperty, Rect position, string title, string fieldName, bool isFlags) where TEnum : Enum
        {
            SerializedProperty fieldProperty = parentProperty.FindPropertyRelative(fieldName);
            TEnum value = (TEnum)Enum.ToObject(typeof(TEnum), fieldProperty.intValue);
            Enum selectedValue = isFlags ?
                EditorGUI.EnumFlagsField(position, title, value)
                : EditorGUI.EnumPopup(position, title, value);
            fieldProperty.intValue = (int)Enum.ToObject(typeof(TEnum), selectedValue);
        }
    }
}
