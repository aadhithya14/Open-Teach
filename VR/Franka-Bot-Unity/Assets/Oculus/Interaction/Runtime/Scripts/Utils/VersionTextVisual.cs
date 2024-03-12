/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using TMPro;
using UnityEngine;
using UnityEngine.Assertions;

namespace Oculus.Interaction
{
    public class VersionTextVisual : MonoBehaviour
    {
        [SerializeField]
        private TextMeshPro _text;

        protected virtual void Start()
        {
            Assert.IsNotNull(_text);
            _text.text = "Version: " + Application.version;
        }

        #region Inject

        public void InjectAllVersionTextVisual(TextMeshPro text)
        {
            InjectText(text);
        }

        public void InjectText(TextMeshPro text)
        {
            _text = text;
        }

        #endregion
    }
}
