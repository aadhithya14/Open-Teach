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

using System;
using Facebook.WitAi;
using Facebook.WitAi.Windows;
using Facebook.WitAi.Utilities;
using Oculus.Voice.Utility;
using UnityEditor;
using UnityEngine;

namespace Oculus.Voice.Windows
{
    public class AboutWindow : WitScriptableWizard
    {
        protected override Texture2D HeaderIcon => VoiceSDKStyles.MainHeader;
        protected override GUIContent Title => VoiceSDKStyles.AboutTitle;
        protected override string ButtonLabel => VoiceSDKStyles.Texts.AboutCloseLabel;
        protected override string ContentSubheaderLabel => string.Empty;

        protected override void LayoutFields()
        {
            WitEditorUI.LayoutKeyLabel(VoiceSDKStyles.Texts.AboutVoiceSdkVersionLabel, VoiceSDKVersion.VERSION);
            WitEditorUI.LayoutKeyLabel(VoiceSDKStyles.Texts.AboutWitSdkVersionLabel, WitRequest.WIT_SDK_VERSION);
            WitEditorUI.LayoutKeyLabel(VoiceSDKStyles.Texts.AboutWitApiVersionLabel, WitRequest.WIT_API_VERSION);

            GUILayout.Space(16);

            if (GUILayout.Button(VoiceSDKStyles.Texts.AboutTutorialButtonLabel, WitStyles.TextButton))
            {
                Application.OpenURL(VoiceSDKStyles.Texts.AboutTutorialButtonUrl);
            }
        }
    }
}
