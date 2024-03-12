/**************************************************************************************************
 * Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.
 *
 * Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 **************************************************************************************************/

using UnityEngine;
using UnityEditor;
using System.Reflection;
using Facebook.WitAi.Data.Configuration;
using Facebook.WitAi.Windows;
using Oculus.Voice.Inspectors;

namespace Oculus.Voice.Windows
{
    public class VoiceApplicationDetailProvider : WitApplicationPropertyDrawer
    {
        // Skip fields if voice sdk app id
        protected override bool ShouldLayoutField(SerializedProperty property, FieldInfo subfield)
        {
            string appID = GetFieldStringValue(property, "id").ToLower();
            if (AppVoiceExperienceWitConfigurationEditor.IsBuiltInConfiguration(appID))
            {
                switch (subfield.Name)
                {
                    case "id":
                    case "createdAt":
                    case "isPrivate":
                        return false;
                }
            }
            return base.ShouldLayoutField(property, subfield);
        }
    }
}
