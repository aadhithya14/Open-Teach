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
    [CustomPropertyDrawer(typeof(ConditionalHideAttribute))]
    public class ConditionalHideDrawer : PropertyDrawer
    {
        bool FulfillsCondition(SerializedProperty property)
        {
            ConditionalHideAttribute hideAttribute = (ConditionalHideAttribute)attribute;

            int index = property.propertyPath.LastIndexOf('.');
            string containerPath = property.propertyPath.Substring(0, index + 1);
            string conditionPath = containerPath + hideAttribute.ConditionalFieldPath;
            SerializedProperty conditionalProperty = property.serializedObject.FindProperty(conditionPath);

            if (conditionalProperty.type == "Enum")
            {
                return conditionalProperty.enumValueIndex == (int)hideAttribute.HideValue;
            }
            if (conditionalProperty.type == "int")
            {
                return conditionalProperty.intValue == (int)hideAttribute.HideValue;
            }
            if (conditionalProperty.type == "float")
            {
                return conditionalProperty.floatValue == (float)hideAttribute.HideValue;
            }
            if (conditionalProperty.type == "string")
            {
                return conditionalProperty.stringValue == (string)hideAttribute.HideValue;
            }
            if (conditionalProperty.type == "double")
            {
                return conditionalProperty.doubleValue == (double)hideAttribute.HideValue;
            }
            if (conditionalProperty.type == "bool")
            {
                return conditionalProperty.boolValue == (bool)hideAttribute.HideValue;
            }

            return conditionalProperty.objectReferenceValue == (object)hideAttribute.HideValue;
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            if (FulfillsCondition(property))
            {
                EditorGUI.PropertyField(position, property, label, true);
            }
        }

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            if (FulfillsCondition(property))
            {
                return EditorGUI.GetPropertyHeight(property, label, true);
            }
            else
            {
                return -EditorGUIUtility.standardVerticalSpacing;
            }
        }
    }
}
