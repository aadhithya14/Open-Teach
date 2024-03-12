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
using Oculus.Interaction.HandGrab.Visuals;

namespace Oculus.Interaction.HandGrab.Editor
{
    public class HandGhostProviderUtils
    {
        public static bool TryGetDefaultProvider(out HandGhostProvider provider)
        {
            provider = null;
            HandGhostProvider[] providers = Resources.FindObjectsOfTypeAll<HandGhostProvider>();
            if (providers != null && providers.Length > 0)
            {
                provider = providers[0];
                return true;
            }

            string[] assets = AssetDatabase.FindAssets($"t:{nameof(HandGhostProvider)}");
            if (assets != null && assets.Length > 0)
            {
                string pathPath = AssetDatabase.GUIDToAssetPath(assets[0]);
                provider = AssetDatabase.LoadAssetAtPath<HandGhostProvider>(pathPath);
            }


            return provider != null;
        }
    }
}
